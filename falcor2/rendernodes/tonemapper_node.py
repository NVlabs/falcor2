# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
import math
from typing import Any, Optional

import numpy as np
import slangpy as spy
from slangpy import CommandEncoder, Device, Module

from falcor2.reflection import UIFlags, reflected, reflected_property
from falcor2.rendergraph import Container, RenderNode
from falcor2.rendergraph import container_torch

HISTOGRAM_BIN_COUNT = 64
HISTOGRAM_MAX_WEIGHT_SCALE = 64
UINT32_MAX = (1 << 32) - 1
TONEMAPPER_MODULE = "falcor2.rendernodes.tonemapper"
MIN_EXPOSURE_COMPENSATION = -32.0
MAX_EXPOSURE_COMPENSATION = 32.0


def _resolve_histogram_weight_scale(pixel_count: int) -> int:
    """Return the largest supported per-pixel weight that cannot overflow uint32."""
    if pixel_count <= 0:
        raise ValueError("Auto-exposure input must not be empty.")
    max_safe_weight_scale = UINT32_MAX // pixel_count
    if max_safe_weight_scale == 0:
        raise ValueError("Auto-exposure input has too many pixels for the histogram.")
    return min(HISTOGRAM_MAX_WEIGHT_SCALE, max_safe_weight_scale)


class AutoExposureMode(IntEnum):
    """Auto-exposure scheduling mode."""

    # Tone map with the stored exposure, then meter the current image and
    # prepare a rate-limited exposure value for the next frame.
    previous_frame = 0

    # Meter the current image and apply its target exposure in the same call.
    current_frame = 1


class TonemappingOperator(IntEnum):
    """Curve used to map exposed scene-linear RGB into display range."""

    aces_film = 0
    reinhard = 1
    clamp = 2
    agx = 3
    lut3d = 4


_OPERATOR_TYPE_NAMES = {
    TonemappingOperator.aces_film: "AcesFilmOperator",
    TonemappingOperator.reinhard: "ReinhardOperator",
    TonemappingOperator.clamp: "ClampOperator",
    TonemappingOperator.agx: "AgXOperator",
}


@reflected
class TonemapperNode(RenderNode):
    def __init__(
        self,
        device: Device,
        operator: TonemappingOperator = TonemappingOperator.aces_film,
    ) -> None:
        super().__init__()
        self._device = device
        self._module = Module(device.load_module(TONEMAPPER_MODULE))
        self._output = None
        self._operator_functions: dict[TonemappingOperator, tuple[Any, Any, Any]] = {}
        self._operator = TonemappingOperator.aces_film
        self._set_operator(operator)
        self._lut: Optional[spy.Texture] = None
        self._lut_sampler: spy.Sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.point,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
        )
        self._lut_min_ev = -10.0
        self._lut_max_ev = 6.5
        self._lut_middle_gray = 0.18
        self._auto_exposure = False
        self._auto_exposure_mode = AutoExposureMode.previous_frame
        self._exposure_compensation = 0.0
        self._exposure_key = 0.18
        self._histogram_log_min = -10.0
        self._histogram_log_max = 20.0
        self._histogram_low_percent = 10.0
        self._histogram_high_percent = 90.0
        self._min_exposure_ev = -16.0
        self._max_exposure_ev = 16.0
        self._adaptation_speed_up = 1.0
        self._adaptation_speed_down = 3.0
        self._luminance_histogram: Optional[spy.Tensor] = None
        self._adapted_exposure_ev: Optional[spy.Buffer] = None
        self._exposure_history_valid = False

    @classmethod
    def create(
        cls,
        device: Device,
        operator: TonemappingOperator = TonemappingOperator.aces_film,
    ) -> "TonemapperNode":
        return cls(device, operator)

    @reflected_property(ui_label="Operator")
    def operator(self) -> TonemappingOperator:
        """Curve used to map exposed scene-linear RGB into display range."""
        return self._operator

    @operator.setter
    def operator(self, value: TonemappingOperator) -> None:
        self._set_operator(value)

    def _set_operator(self, value: TonemappingOperator) -> None:
        operator = TonemappingOperator(value)
        functions = self._operator_functions.get(operator)
        if functions is None:
            if operator == TonemappingOperator.lut3d:
                functions = (
                    self._module.require_function("tonemapper_apply_lut3d"),
                    self._module.require_function("tonemapper_apply_lut3d_auto"),
                    None,
                )
            else:
                type_name = _OPERATOR_TYPE_NAMES[operator]
                functions = (
                    self._module.require_function(f"tonemapper_apply<{type_name}>"),
                    self._module.require_function(f"tonemapper_apply_auto<{type_name}>"),
                    self._module.require_function(f"tonemapper_apply_auto_histogram<{type_name}>"),
                )
            self._operator_functions[operator] = functions

        self._operator = operator
        (
            self._tonemap_func,
            self._tonemap_auto_func,
            self._tonemap_auto_histogram_func,
        ) = functions

    @property
    def lut(self) -> Optional[spy.Texture]:
        """Current linear-RGB 3D LUT, with red, green, blue mapped to X, Y, Z."""
        return self._lut

    @lut.setter
    def lut(self, value: np.ndarray | spy.Texture | None) -> None:
        """Set a cubic LUT from an array in (blue, green, red, channel) order or a texture."""
        if value is None:
            self._lut = None
            return
        if isinstance(value, np.ndarray):
            if value.ndim != 4 or value.shape[0] < 2 or value.shape[:3] != (value.shape[0],) * 3:
                raise ValueError("lut must have shape (N, N, N, 3 or 4) with N >= 2.")
            if value.shape[3] not in (3, 4):
                raise ValueError("lut must have three or four channels.")
            if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
                raise ValueError("lut must contain finite floating-point values.")
            if np.max(np.abs(value)) > np.finfo(np.float32).max:
                raise ValueError("lut values must fit in float32.")
            rgba = np.empty((*value.shape[:3], 4), dtype=np.float32)
            rgba[..., :3] = value[..., :3]
            rgba[..., 3] = value[..., 3] if value.shape[3] == 4 else 1.0
            size = value.shape[0]
            value = self._device.create_texture(
                type=spy.TextureType.texture_3d,
                format=spy.Format.rgba32_float,
                width=size,
                height=size,
                depth=size,
                mip_count=1,
                usage=spy.TextureUsage.shader_resource,
                data=rgba,
            )
        if not isinstance(value, spy.Texture):
            raise TypeError("lut must be a NumPy array, a 3D texture, or None.")
        if (
            value.device != self._device
            or value.type != spy.TextureType.texture_3d
            or value.format != spy.Format.rgba32_float
            or value.width < 2
            or value.width != value.height
            or value.width != value.depth
        ):
            raise ValueError(
                "lut texture must be a same-device cubic rgba32_float 3D texture with N >= 2."
            )
        self._lut = value

    @reflected_property(value_range=(-64.0, 64.0), ui_label="LUT minimum EV", ui_group="3D LUT")
    def lut_min_ev(self) -> float:
        """Log2 exposure relative to middle gray at the first LUT voxel."""
        return self._lut_min_ev

    @lut_min_ev.setter
    def lut_min_ev(self, value: float) -> None:
        if not math.isfinite(value) or value < -64.0 or value >= self._lut_max_ev:
            raise ValueError("lut_min_ev must be finite, at least -64, and below lut_max_ev.")
        self._lut_min_ev = value

    @reflected_property(value_range=(-64.0, 64.0), ui_label="LUT maximum EV", ui_group="3D LUT")
    def lut_max_ev(self) -> float:
        """Log2 exposure relative to middle gray at the last LUT voxel."""
        return self._lut_max_ev

    @lut_max_ev.setter
    def lut_max_ev(self, value: float) -> None:
        if not math.isfinite(value) or value > 64.0 or value <= self._lut_min_ev:
            raise ValueError("lut_max_ev must be finite, at most 64, and above lut_min_ev.")
        self._lut_max_ev = value

    @reflected_property(value_range=(0.001, 1.0), ui_label="LUT middle gray", ui_group="3D LUT")
    def lut_middle_gray(self) -> float:
        """Positive scene-linear value used as the zero-EV LUT pivot."""
        return self._lut_middle_gray

    @lut_middle_gray.setter
    def lut_middle_gray(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("lut_middle_gray must be finite and positive.")
        self._lut_middle_gray = value

    @reflected_property(
        value_range=(MIN_EXPOSURE_COMPENSATION, MAX_EXPOSURE_COMPENSATION),
        ui_label="Exposure compensation (EV)",
        ui_drag_speed=0.1,
    )
    def exposure_compensation(self) -> float:
        """Manual exposure offset in EV stops."""
        return self._exposure_compensation

    @exposure_compensation.setter
    def exposure_compensation(self, value: float) -> None:
        self._exposure_compensation = min(
            max(value, MIN_EXPOSURE_COMPENSATION), MAX_EXPOSURE_COMPENSATION
        )

    @reflected_property(ui_label="Auto exposure")
    def auto_exposure(self) -> bool:
        """Whether automatic exposure metering is enabled."""
        return self._auto_exposure

    @auto_exposure.setter
    def auto_exposure(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("auto_exposure must be a bool.")
        if value and not self._auto_exposure:
            self._exposure_history_valid = False
        self._auto_exposure = value

    @reflected_property(
        ui_label="Mode",
        ui_group="Auto exposure",
        ui_enable_if=lambda tonemapper: tonemapper.auto_exposure,
    )
    def auto_exposure_mode(self) -> AutoExposureMode:
        """Whether exposure is metered from the current or previous frame."""
        return self._auto_exposure_mode

    @auto_exposure_mode.setter
    def auto_exposure_mode(self, value: AutoExposureMode) -> None:
        self._auto_exposure_mode = AutoExposureMode(value)

    @reflected_property(
        value_range=(0.001, 1.0),
        ui_label="Middle-gray key",
        ui_group="Auto exposure",
        ui_drag_speed=0.001,
        ui_enable_if=lambda tonemapper: tonemapper.auto_exposure,
    )
    def exposure_key(self) -> float:
        """Middle-gray key used by automatic exposure."""
        return self._exposure_key

    @exposure_key.setter
    def exposure_key(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("exposure_key must be greater than zero.")
        self._exposure_key = value

    @reflected_property(
        value_range=(-32.0, 32.0),
        ui_label="Log luminance minimum",
        ui_group="Auto exposure/Histogram",
        ui_drag_speed=0.1,
        ui_flags=UIFlags.advanced,
        ui_enable_if=lambda tonemapper: tonemapper.auto_exposure,
    )
    def histogram_log_min(self) -> float:
        """Lower log2-luminance limit represented by the histogram."""
        return self._histogram_log_min

    @histogram_log_min.setter
    def histogram_log_min(self, value: float) -> None:
        if value >= self._histogram_log_max:
            raise ValueError("histogram_log_min must be less than histogram_log_max.")
        self._histogram_log_min = value

    @reflected_property(
        value_range=(-32.0, 32.0),
        ui_label="Log luminance maximum",
        ui_group="Auto exposure/Histogram",
        ui_drag_speed=0.1,
        ui_flags=UIFlags.advanced,
        ui_enable_if=lambda tonemapper: tonemapper.auto_exposure,
    )
    def histogram_log_max(self) -> float:
        """Upper log2-luminance limit represented by the histogram."""
        return self._histogram_log_max

    @histogram_log_max.setter
    def histogram_log_max(self, value: float) -> None:
        if value <= self._histogram_log_min:
            raise ValueError("histogram_log_max must be greater than histogram_log_min.")
        self._histogram_log_max = value

    @reflected_property(
        value_range=(0.0, 100.0),
        ui_label="Low percentile",
        ui_group="Auto exposure/Histogram",
        ui_drag_speed=0.1,
        ui_flags=UIFlags.advanced,
        ui_enable_if=lambda tonemapper: tonemapper.auto_exposure,
    )
    def histogram_low_percent(self) -> float:
        """Cumulative dark-tail percentage excluded from metering."""
        return self._histogram_low_percent

    @histogram_low_percent.setter
    def histogram_low_percent(self, value: float) -> None:
        if value < 0.0 or value > 100.0:
            raise ValueError("histogram_low_percent must be between 0 and 100.")
        if value > self._histogram_high_percent:
            raise ValueError("histogram_low_percent must not exceed histogram_high_percent.")
        self._histogram_low_percent = value

    @reflected_property(
        value_range=(0.0, 100.0),
        ui_label="High percentile",
        ui_group="Auto exposure/Histogram",
        ui_drag_speed=0.1,
        ui_flags=UIFlags.advanced,
        ui_enable_if=lambda tonemapper: tonemapper.auto_exposure,
    )
    def histogram_high_percent(self) -> float:
        """Cumulative bright-tail percentage retained by metering."""
        return self._histogram_high_percent

    @histogram_high_percent.setter
    def histogram_high_percent(self, value: float) -> None:
        if value < 0.0 or value > 100.0:
            raise ValueError("histogram_high_percent must be between 0 and 100.")
        if value < self._histogram_low_percent:
            raise ValueError("histogram_high_percent must not be less than histogram_low_percent.")
        self._histogram_high_percent = value

    @reflected_property(
        value_range=(-32.0, 32.0),
        ui_label="Minimum exposure (EV)",
        ui_group="Auto exposure",
        ui_drag_speed=0.1,
        ui_enable_if=lambda tonemapper: tonemapper.auto_exposure,
    )
    def min_exposure_ev(self) -> float:
        """Minimum automatic exposure in EV stops."""
        return self._min_exposure_ev

    @min_exposure_ev.setter
    def min_exposure_ev(self, value: float) -> None:
        if value > self._max_exposure_ev:
            raise ValueError("min_exposure_ev must not exceed max_exposure_ev.")
        if value != self._min_exposure_ev:
            self._exposure_history_valid = False
        self._min_exposure_ev = value

    @reflected_property(
        value_range=(-32.0, 32.0),
        ui_label="Maximum exposure (EV)",
        ui_group="Auto exposure",
        ui_drag_speed=0.1,
        ui_enable_if=lambda tonemapper: tonemapper.auto_exposure,
    )
    def max_exposure_ev(self) -> float:
        """Maximum automatic exposure in EV stops."""
        return self._max_exposure_ev

    @max_exposure_ev.setter
    def max_exposure_ev(self, value: float) -> None:
        if value < self._min_exposure_ev:
            raise ValueError("max_exposure_ev must not be less than min_exposure_ev.")
        if value != self._max_exposure_ev:
            self._exposure_history_valid = False
        self._max_exposure_ev = value

    @reflected_property(
        value_range=(0.0, 16.0),
        ui_label="Brightening speed (EV/s)",
        ui_group="Auto exposure",
        ui_drag_speed=0.1,
        ui_enable_if=lambda tonemapper: tonemapper.auto_exposure
        and tonemapper.auto_exposure_mode == AutoExposureMode.previous_frame,
    )
    def adaptation_speed_up(self) -> float:
        """Maximum brightening speed in EV per second."""
        return self._adaptation_speed_up

    @adaptation_speed_up.setter
    def adaptation_speed_up(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("adaptation_speed_up must be non-negative.")
        self._adaptation_speed_up = value

    @reflected_property(
        value_range=(0.0, 16.0),
        ui_label="Darkening speed (EV/s)",
        ui_group="Auto exposure",
        ui_drag_speed=0.1,
        ui_enable_if=lambda tonemapper: tonemapper.auto_exposure
        and tonemapper.auto_exposure_mode == AutoExposureMode.previous_frame,
    )
    def adaptation_speed_down(self) -> float:
        """Maximum darkening speed in EV per second."""
        return self._adaptation_speed_down

    @adaptation_speed_down.setter
    def adaptation_speed_down(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("adaptation_speed_down must be non-negative.")
        self._adaptation_speed_down = value

    def reset(self, cmd: Optional[CommandEncoder] = None) -> None:
        """Invalidate temporal exposure history without submitting GPU work."""
        del cmd
        self._exposure_history_valid = False

    def _get_output(self, input: Any):
        self._output = Container.create_temp_like(self._device, input, current=self._output)
        return self._output

    def _validate_auto_exposure_input(self, input: Any) -> int:
        format_value, dims = Container.format_and_dims(input)
        if len(dims) != 2:
            raise ValueError("Auto-exposure input must be a two-dimensional image.")
        height, width = int(dims[0]), int(dims[1])
        if height <= 0 or width <= 0:
            raise ValueError("Auto-exposure input must not be empty.")

        format_info = spy.get_format_info(format_value)
        channels = format_info.channel_count
        if not format_info.is_float_format() or channels not in (3, 4):
            raise ValueError("Auto-exposure input must be floating-point RGB or RGBA.")

        if isinstance(input, (spy.Tensor, spy.Texture)) and input.device != self._device:
            raise ValueError("Auto-exposure input must use the TonemapperNode device.")
        if container_torch.is_torch_tensor(input):
            if not input.is_cuda or self._device.desc.type != spy.DeviceType.cuda:
                raise ValueError(
                    "Torch auto-exposure input must use the TonemapperNode CUDA device."
                )

        return _resolve_histogram_weight_scale(height * width)

    def _ensure_auto_exposure_resources(self) -> None:
        if self._luminance_histogram is None:
            self._luminance_histogram = spy.Tensor.empty(
                self._device, shape=(HISTOGRAM_BIN_COUNT,), dtype="uint"
            )
        if self._adapted_exposure_ev is None:
            self._adapted_exposure_ev = self._device.create_buffer(
                size=4,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )

    def _append_meter(
        self,
        runtime_input: Any,
        histogram_weight_scale: int,
        cmd: CommandEncoder,
    ) -> None:
        assert self._luminance_histogram is not None
        self._module.tonemapper_accumulate_luminance_histogram(
            color=runtime_input,
            luminance_histogram=self._luminance_histogram.storage,
            histogram_weight_scale=histogram_weight_scale,
            histogram_log_min=self._histogram_log_min,
            histogram_log_max=self._histogram_log_max,
            _append_to=cmd,
        )

    def _append_exposure_update(
        self,
        delta_time: float,
        apply_immediately: bool,
        cmd: CommandEncoder,
    ) -> None:
        assert self._luminance_histogram is not None
        assert self._adapted_exposure_ev is not None
        self._module.tonemapper_update_auto_exposure(
            luminance_histogram=self._luminance_histogram.storage,
            adapted_exposure_ev=self._adapted_exposure_ev,
            histogram_log_min=self._histogram_log_min,
            histogram_log_max=self._histogram_log_max,
            histogram_low_percent=self._histogram_low_percent,
            histogram_high_percent=self._histogram_high_percent,
            exposure_key=self._exposure_key,
            exposure_compensation=self._exposure_compensation,
            min_exposure_ev=self._min_exposure_ev,
            max_exposure_ev=self._max_exposure_ev,
            adaptation_speed_up=self._adaptation_speed_up,
            adaptation_speed_down=self._adaptation_speed_down,
            delta_time=delta_time,
            apply_immediately=apply_immediately,
            _append_to=cmd,
        )

    def _lut_parameters(self) -> dict[str, Any]:
        if self._lut is None:
            raise ValueError("lut must be set before rendering with the lut3d operator.")
        return {
            "lut": self._lut,
            "lut_sampler": self._lut_sampler,
            "lut_min_ev": self._lut_min_ev,
            "lut_max_ev": self._lut_max_ev,
            "lut_middle_gray": self._lut_middle_gray,
        }

    def _append_tonemap(
        self,
        runtime_input: Any,
        runtime_output: Any,
        cmd: Optional[CommandEncoder],
    ) -> None:
        lut_parameters = (
            self._lut_parameters() if self._operator == TonemappingOperator.lut3d else {}
        )

        if self._auto_exposure:
            assert self._adapted_exposure_ev is not None
            self._tonemap_auto_func(
                color=runtime_input,
                adapted_exposure_ev=self._adapted_exposure_ev,
                **lut_parameters,
                _result=runtime_output,
                _append_to=cmd,
            )
        else:
            self._tonemap_func(
                color=runtime_input,
                exposure_ev=self._exposure_compensation,
                **lut_parameters,
                _result=runtime_output,
                _append_to=cmd,
            )

    def _append_fused_tonemap_and_meter(
        self,
        runtime_input: Any,
        runtime_output: Any,
        histogram_weight_scale: int,
        cmd: CommandEncoder,
    ) -> None:
        assert self._luminance_histogram is not None
        assert self._adapted_exposure_ev is not None
        if self._operator == TonemappingOperator.lut3d:
            self._append_tonemap(runtime_input, runtime_output, cmd)
            self._append_meter(runtime_input, histogram_weight_scale, cmd)
            return
        self._tonemap_auto_histogram_func(
            color=runtime_input,
            adapted_exposure_ev=self._adapted_exposure_ev,
            luminance_histogram=self._luminance_histogram.storage,
            histogram_weight_scale=histogram_weight_scale,
            histogram_log_min=self._histogram_log_min,
            histogram_log_max=self._histogram_log_max,
            mapped_color=runtime_output,
            _append_to=cmd,
        )

    def _exec(
        self,
        input: Any,
        cmd: Optional[CommandEncoder] = None,
        delta_time: float = 1.0 / 60.0,
    ):
        output = self._get_output(input)
        runtime_input = Container.to_render_layout(input)
        runtime_output = Container.to_render_layout(output)

        if not self._auto_exposure:
            self._append_tonemap(runtime_input, runtime_output, cmd)
            return output

        if delta_time < 0.0:
            raise ValueError("delta_time must be non-negative.")

        histogram_weight_scale = self._validate_auto_exposure_input(input)
        self._ensure_auto_exposure_resources()
        assert self._luminance_histogram is not None
        owns_cmd = cmd is None
        command_encoder = cmd or self._device.create_command_encoder()
        apply_immediately = (
            self._auto_exposure_mode == AutoExposureMode.current_frame
            or not self._exposure_history_valid
        )

        self._luminance_histogram.clear(command_encoder)
        if apply_immediately:
            self._append_meter(
                runtime_input,
                histogram_weight_scale,
                command_encoder,
            )
            self._append_exposure_update(
                delta_time,
                apply_immediately=True,
                cmd=command_encoder,
            )
            self._append_tonemap(
                runtime_input,
                runtime_output,
                command_encoder,
            )
        else:
            self._append_fused_tonemap_and_meter(
                runtime_input,
                runtime_output,
                histogram_weight_scale,
                command_encoder,
            )
            self._append_exposure_update(
                delta_time,
                apply_immediately=False,
                cmd=command_encoder,
            )

        self._exposure_history_valid = True
        if owns_cmd:
            self._device.submit_command_buffer(command_encoder.finish())
        return output
