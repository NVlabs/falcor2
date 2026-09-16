# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Optional

import slangpy as spy

from falcor2.reflection import reflected, reflected_property
from falcor2.rendergraph import Container, ContainerSpec, RenderNode
from falcor2.rendergraph.image_format import channel_count


_ACCUMULATOR_MODULE = "falcor2.rendernodes.accumulator"
_ACCUMULATOR_DOUBLE_MODULE = "falcor2.rendernodes.accumulator_double"
_MAX_SAMPLE_COUNT = 0xFFFFFFFF
_DEFAULT_MAX_SAMPLE_LUMINANCE = 1000.0


class AccumulatorPrecision(IntEnum):
    """Internal precision used for temporal accumulation history."""

    double = 0
    single = 1
    single_compensated = 2


@reflected
class AccumulatorNode(RenderNode):
    def __init__(
        self,
        device: spy.Device,
        precision: AccumulatorPrecision = AccumulatorPrecision.single_compensated,
        *,
        sanitize_non_finite: bool = True,
        clamp_sample_luminance: bool = False,
        max_sample_luminance: float = _DEFAULT_MAX_SAMPLE_LUMINANCE,
    ) -> None:
        super().__init__()
        self._device = device
        self._single_module: spy.Module | None = None
        self._compensated_session: Any | None = None
        self._compensated_module: spy.Module | None = None
        self._double_module: spy.Module | None = None

        self._history: spy.Tensor | None = None
        self._input_spec: ContainerSpec | None = None
        self._output: Any = None
        self._sample_count = 0
        self._sanitize_non_finite = bool(sanitize_non_finite)
        self._clamp_sample_luminance = bool(clamp_sample_luminance)
        self._max_sample_luminance = self._validate_max_sample_luminance(max_sample_luminance)

        # Kept for source compatibility. Repairing these properties is tracked
        # separately in accumulator backlog item ACC-001.
        self._width = 0
        self._height = 0

        requested_precision = AccumulatorPrecision(precision)
        update_func, history_dtype = self._resolve_precision(requested_precision)
        self._precision = requested_precision
        self._update_func = update_func
        self._history_dtype = history_dtype

    @classmethod
    def create(
        cls,
        device: spy.Device,
        precision: AccumulatorPrecision = AccumulatorPrecision.single_compensated,
        *,
        sanitize_non_finite: bool = True,
        clamp_sample_luminance: bool = False,
        max_sample_luminance: float = _DEFAULT_MAX_SAMPLE_LUMINANCE,
    ) -> "AccumulatorNode":
        return cls(
            device,
            precision,
            sanitize_non_finite=sanitize_non_finite,
            clamp_sample_luminance=clamp_sample_luminance,
            max_sample_luminance=max_sample_luminance,
        )

    @staticmethod
    def _validate_max_sample_luminance(value: float) -> float:
        result = float(value)
        if not result >= 0.0:
            raise ValueError("max_sample_luminance must be non-negative.")
        return result

    @reflected_property(ui_label="Precision")
    def precision(self) -> AccumulatorPrecision:
        return self._precision

    @precision.setter
    def precision(self, value: AccumulatorPrecision) -> None:
        precision = AccumulatorPrecision(value)
        if precision == self._precision:
            return

        update_func, history_dtype = self._resolve_precision(precision)

        self._precision = precision
        self._update_func = update_func
        self._history_dtype = history_dtype
        self._release_history()

    @reflected_property(ui_label="Sanitize non-finite samples")
    def sanitize_non_finite(self) -> bool:
        """Whether non-finite input samples are replaced with zero."""
        return self._sanitize_non_finite

    @sanitize_non_finite.setter
    def sanitize_non_finite(self, value: bool) -> None:
        enabled = bool(value)
        if enabled != self._sanitize_non_finite:
            self._sanitize_non_finite = enabled
            self.reset()

    @reflected_property(ui_label="Clamp sample luminance")
    def clamp_sample_luminance(self) -> bool:
        """Whether input samples are clamped to a finite maximum luminance."""
        return self._clamp_sample_luminance

    @clamp_sample_luminance.setter
    def clamp_sample_luminance(self, value: bool) -> None:
        enabled = bool(value)
        if enabled != self._clamp_sample_luminance:
            self._clamp_sample_luminance = enabled
            self.reset()

    @reflected_property(
        value_range=(0.0, math.inf),
        ui_label="Maximum sample luminance",
        ui_drag_speed=1.0,
        ui_enable_if=lambda accumulator: accumulator.clamp_sample_luminance,
    )
    def max_sample_luminance(self) -> float:
        """Configured maximum luminance of one input sample."""
        return self._max_sample_luminance

    @max_sample_luminance.setter
    def max_sample_luminance(self, value: float) -> None:
        limit = self._validate_max_sample_luminance(value)
        if limit != self._max_sample_luminance:
            self._max_sample_luminance = limit
            self.reset()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _supports_double(self) -> bool:
        return self._device.has_feature(spy.Feature.double)

    def _load_single_module(self) -> spy.Module:
        if self._single_module is None:
            self._single_module = spy.Module(self._device.load_module(_ACCUMULATOR_MODULE))
        return self._single_module

    def _load_compensated_module(self) -> spy.Module:
        if self._compensated_module is None:
            session = self._device.create_slang_session(
                {
                    "include_paths": self._device.slang_session.desc.compiler_options.include_paths,
                    "floating_point_mode": spy.SlangFloatingPointMode.precise,
                }
            )
            self._compensated_session = session
            self._compensated_module = spy.Module(session.load_module(_ACCUMULATOR_MODULE))
        return self._compensated_module

    def _load_double_module(self) -> spy.Module:
        if self._double_module is None:
            self._double_module = spy.Module(self._device.load_module(_ACCUMULATOR_DOUBLE_MODULE))
        return self._double_module

    def _resolve_precision(self, precision: AccumulatorPrecision) -> tuple[Any, Any]:
        if precision == AccumulatorPrecision.single:
            module = self._load_single_module()
            return module.accumulator_update_single, "float4"
        if precision == AccumulatorPrecision.single_compensated:
            module = self._load_compensated_module()
            history_dtype = module.layout.require_type_by_name("AccumHistory4")
            return module.accumulator_update_single_compensated, history_dtype
        if not self._supports_double():
            raise RuntimeError("Double accumulation is not supported by this device.")
        module = self._load_double_module()
        return module.accumulator_update_double, "double4"

    def _get_input_spec(self, input: Any) -> ContainerSpec:
        if isinstance(input, (spy.Tensor, spy.Texture)) and input.device != self._device:
            raise ValueError("Accumulator input must use the node's device.")

        spec = ContainerSpec.from_container(input)
        if (
            not isinstance(spec.dims, tuple)
            or len(spec.dims) != 2
            or channel_count(spec.format) != 4
        ):
            raise ValueError("Accumulator input must be a two-dimensional RGBA image.")
        return spec

    def _release_history(self) -> None:
        self._history = None
        self._input_spec = None
        self._sample_count = 0

    def _ensure_history(self, input_spec: ContainerSpec) -> spy.Tensor:
        if input_spec != self._input_spec:
            assert isinstance(input_spec.dims, tuple)
            self._history = spy.Tensor.empty(
                self._device,
                shape=input_spec.dims,
                dtype=self._history_dtype,
            )
            self._input_spec = input_spec
            self._sample_count = 0
        assert self._history is not None
        return self._history

    def reset(self, cmd: Optional[spy.CommandEncoder] = None) -> None:
        del cmd
        self._sample_count = 0

    def _get_output(self, input: Any) -> Any:
        self._output = Container.create_temp_like(self._device, input, current=self._output)
        return self._output

    def _exec(self, input: Any, cmd: Optional[spy.CommandEncoder] = None) -> Any:
        if self._sample_count == _MAX_SAMPLE_COUNT:
            raise OverflowError("Accumulator sample count reached UINT32_MAX.")

        input_spec = self._get_input_spec(input)
        history = self._ensure_history(input_spec)
        output = self._get_output(input)
        render_input = Container.to_render_layout(input)
        render_output = Container.to_render_layout(output)

        self._update_func(
            render_input,
            history,
            self._sample_count,
            self._sanitize_non_finite,
            self._max_sample_luminance if self._clamp_sample_luminance else math.inf,
            render_output,
            _append_to=cmd,
        )

        self._sample_count += 1
        return output
