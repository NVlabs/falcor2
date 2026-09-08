# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Optional

import slangpy as spy

from falcor2.reflection import reflected, reflected_property
from falcor2.rendergraph import Container, ContainerSpec, RenderNode


_REWEIGHTING_ACCUMULATOR_MODULE = "falcor2.rendernodes.reweighting_accumulator"
_FLOAT_RGBA_FORMATS = frozenset((spy.Format.rgba16_float, spy.Format.rgba32_float))
_MAX_BIN_COUNT = 43
_MIN_NORMAL_FLOAT32 = 2.0**-126


class ReweightingAccumulatorOutput(IntEnum):
    """Output reconstructed from a reweighting accumulation history."""

    unbiased = 0
    reweighted = 1


@reflected
class ReweightingAccumulatorNode(RenderNode):
    """Accumulate linear radiance into fixed ``1, 8, 64, ...`` luminance bins.

    Implements "Reweighting Firefly Samples for Improved Finite-Sample Monte
    Carlo Estimates" by Tobias Zirr, Johannes Hanika, and Carsten Dachsbacher
    (2018):
    https://cg.ivd.kit.edu/publications/2018/rwmc/reweighting-fireflies-preprint.pdf

    A finite path-sample clamp changes the input distribution. Consequently, the
    unbiased output is unbiased only with respect to the samples received here.
    Paper-faithful comparisons should disable any upstream sample clamp.
    """

    def __init__(
        self,
        device: spy.Device,
        *,
        bin_count: int = 8,
        kappa: float = 1.0,
        kappa_min: float = 1.0,
        output_mode: ReweightingAccumulatorOutput = ReweightingAccumulatorOutput.reweighted,
        sanitize_non_finite: bool = True,
    ) -> None:
        super().__init__()

        self._device = device
        self._bin_count = self._validate_bin_count(bin_count)
        self._kappa = self._validate_kappa(kappa)
        self._kappa_min = self._validate_kappa_min(kappa_min)
        self._output_mode = ReweightingAccumulatorOutput(output_mode)
        self._sanitize_non_finite = bool(sanitize_non_finite)

        module = spy.Module(self._device.load_module(_REWEIGHTING_ACCUMULATOR_MODULE))
        self._accumulate_func = module.reweighting_accumulator_accumulate
        self._resolve_func = module.reweighting_accumulator_resolve

        self._history: spy.Tensor | None = None
        self._overflow_count: spy.Tensor | None = None
        self._input_spec: ContainerSpec | None = None
        self._last_input: Any = None
        self._output: Any = None
        self._sample_count = 0

    @classmethod
    def create(
        cls,
        device: spy.Device,
        *,
        bin_count: int = 8,
        kappa: float = 1.0,
        kappa_min: float = 1.0,
        output_mode: ReweightingAccumulatorOutput = ReweightingAccumulatorOutput.reweighted,
        sanitize_non_finite: bool = True,
    ) -> "ReweightingAccumulatorNode":
        return cls(
            device,
            bin_count=bin_count,
            kappa=kappa,
            kappa_min=kappa_min,
            output_mode=output_mode,
            sanitize_non_finite=sanitize_non_finite,
        )

    @staticmethod
    def _validate_bin_count(value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("bin_count must be an integer.")
        if value < 2:
            raise ValueError("bin_count must be at least 2.")
        if value > _MAX_BIN_COUNT:
            raise ValueError("bin_count produces a non-finite FP32 bin anchor.")
        return value

    @staticmethod
    def _validate_kappa(value: float) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("kappa must be finite and non-negative.")
        return result

    @staticmethod
    def _validate_kappa_min(value: float) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("kappa_min must be finite and non-negative.")
        return result

    @reflected_property(value_range=(2, _MAX_BIN_COUNT), ui_label="Bin count")
    def bin_count(self) -> int:
        """Number of finite bins; anchors are fixed to ``8 ** bin_index``."""
        return self._bin_count

    @bin_count.setter
    def bin_count(self, value: int) -> None:
        bin_count = self._validate_bin_count(value)
        if bin_count != self._bin_count:
            self._bin_count = bin_count
            self._history = None
            self._overflow_count = None
            self._input_spec = None
            self.reset()

    @reflected_property(
        value_range=(0.0, math.inf),
        ui_label="Kappa",
    )
    def kappa(self) -> float:
        """Unitless variance-versus-bias control, defaulting to 1."""
        return self._kappa

    @kappa.setter
    def kappa(self, value: float) -> None:
        self._kappa = self._validate_kappa(value)

    @reflected_property(
        value_range=(0.0, math.inf),
        ui_label="Minimum kappa",
    )
    def kappa_min(self) -> float:
        """Minimum trusted soft occurrence count, defaulting to 1."""
        return self._kappa_min

    @kappa_min.setter
    def kappa_min(self, value: float) -> None:
        self._kappa_min = self._validate_kappa_min(value)

    @reflected_property(ui_label="Output")
    def output_mode(self) -> ReweightingAccumulatorOutput:
        """Default reconstruction selected by :meth:`resolve`."""
        return self._output_mode

    @output_mode.setter
    def output_mode(self, value: ReweightingAccumulatorOutput) -> None:
        self._output_mode = ReweightingAccumulatorOutput(value)

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

    def _get_input_spec(self, input: Any) -> ContainerSpec:
        spec = ContainerSpec.from_container(input)
        if spec.format not in _FLOAT_RGBA_FORMATS:
            raise ValueError("Reweighting accumulator input must be floating-point RGBA.")
        return spec

    def _ensure_history(self, input_spec: ContainerSpec) -> tuple[spy.Tensor, spy.Tensor]:
        if input_spec != self._input_spec:
            self._history = spy.Tensor.empty(
                self._device,
                shape=(*input_spec.dims, self._bin_count + 1),
                dtype="float4",
            )
            self._overflow_count = spy.Tensor.empty(self._device, shape=(1,), dtype="uint")
            self._input_spec = input_spec
            self._sample_count = 0

        return self._history, self._overflow_count

    def _append_accumulate(self, input: Any, command_encoder: spy.CommandEncoder) -> None:
        input_spec = self._get_input_spec(input)
        history, overflow_count = self._ensure_history(input_spec)

        if self._sample_count == 0:
            history.clear(command_encoder)
            overflow_count.clear(command_encoder)

        self._accumulate_func(
            input=Container.to_render_layout(input),
            history=history,
            overflow_count=overflow_count,
            bin_count=self._bin_count,
            sanitize_non_finite=self._sanitize_non_finite,
            _append_to=command_encoder,
        )
        self._last_input = input
        self._sample_count += 1

    def accumulate(self, input: Any, cmd: Optional[spy.CommandEncoder] = None) -> None:
        """Append one non-negative linear RGBA sample to the current history."""
        owns_cmd = cmd is None
        command_encoder = cmd or self._device.create_command_encoder()
        self._append_accumulate(input, command_encoder)
        if owns_cmd:
            self._device.submit_command_buffer(command_encoder.finish())

    def _get_output(self, output_like: Any) -> Any:
        self._output = Container.create_temp_like(self._device, output_like, current=self._output)
        return self._output

    def _append_resolve(
        self,
        output_like: Any | None,
        command_encoder: spy.CommandEncoder,
        output_mode: ReweightingAccumulatorOutput | None,
    ) -> Any:
        if self._sample_count == 0 or self._history is None or self._input_spec is None:
            raise RuntimeError(
                "Cannot resolve a reweighting accumulator before accumulating a sample."
            )

        template = self._last_input if output_like is None else output_like
        output_spec = self._get_input_spec(template)
        if output_spec.dims != self._input_spec.dims:
            raise ValueError(
                "Reweighting accumulator resolve output dimensions must match history."
            )

        mode = (
            self._output_mode if output_mode is None else ReweightingAccumulatorOutput(output_mode)
        )
        output = self._get_output(template)
        self._resolve_func(
            pixel=spy.grid(output_spec.dims),
            history=self._history.storage,
            image_shape=spy.uint2(*output_spec.dims),
            bin_count=self._bin_count,
            sample_count=self._sample_count,
            kappa=max(self._kappa, _MIN_NORMAL_FLOAT32),
            kappa_min=self._kappa_min,
            output_mode=int(mode),
            output=Container.to_render_layout(output),
            _append_to=command_encoder,
        )
        return output

    def resolve(
        self,
        output_like: Any | None = None,
        cmd: Optional[spy.CommandEncoder] = None,
        output_mode: ReweightingAccumulatorOutput | None = None,
    ) -> Any:
        """Resolve the current history without changing it."""
        owns_cmd = cmd is None
        command_encoder = cmd or self._device.create_command_encoder()
        output = self._append_resolve(output_like, command_encoder, output_mode)
        if owns_cmd:
            self._device.submit_command_buffer(command_encoder.finish())
        return output

    def _exec(self, input: Any, cmd: Optional[spy.CommandEncoder] = None) -> Any:
        """Append one sample and resolve the selected output mode."""
        owns_cmd = cmd is None
        command_encoder = cmd or self._device.create_command_encoder()
        self._append_accumulate(input, command_encoder)
        output = self._append_resolve(input, command_encoder, None)
        if owns_cmd:
            self._device.submit_command_buffer(command_encoder.finish())
        return output

    def reset(self, cmd: Optional[spy.CommandEncoder] = None) -> None:
        """Reset host state; GPU history is cleared before the next accumulation."""
        del cmd
        self._sample_count = 0
        self._last_input = None

    def read_overflow_count(self) -> int:
        """Read the number of samples excluded by the finite cascade range."""
        if self._sample_count == 0 or self._overflow_count is None:
            return 0
        self._device.wait()
        return int(self._overflow_count.to_numpy()[0])
