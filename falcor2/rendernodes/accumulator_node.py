# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from slangpy import CommandEncoder, Device, Module

from falcor2.rendergraph import RenderNode, Container


class AccumulatorNode(RenderNode):
    def __init__(self, device: Device):
        super().__init__()
        self._device = device
        self._module = Module(device.load_module("falcor2.utils"))
        self._reset_func = self._module.accumulator_reset
        self._update_and_output_func = self._module.accumulator_update_and_output
        self._accumulator = None
        self._output = None
        self._width = 0
        self._height = 0

    @classmethod
    def create(cls, device: Device) -> "AccumulatorNode":
        return cls(device)

    def _ensure_buffers(self, input: Any, cmd: Optional[CommandEncoder] = None):
        if not Container.is_like(self._accumulator, input):
            self._accumulator = Container.empty_like(input)
            self.reset(cmd=cmd)

    def reset(self, cmd: Optional[CommandEncoder] = None):
        if self._accumulator is not None:
            self._reset_func(Container.to_render_layout(self._accumulator), _append_to=cmd)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _get_output(self, input: Any):
        self._output = Container.create_temp_like(self._device, input, current=self._output)
        return self._output

    def forward(self, input: Any, cmd: Optional[CommandEncoder] = None):
        self._ensure_buffers(input, cmd=cmd)
        output = self._get_output(input)
        self._update_and_output_func(
            Container.to_render_layout(input),
            Container.to_render_layout(self._accumulator),
            Container.to_render_layout(output),
            _append_to=cmd,
        )
        self._output = output
        return self._output
