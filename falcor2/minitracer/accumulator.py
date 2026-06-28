# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from slangpy import Device, float4, Tensor, Module, CommandEncoder


class Accumulator:
    """
    Helper class to accumulate rendered frames.
    """

    def __init__(self, device: Device, width: int, height: int):
        super().__init__()
        self._device = device
        self._width = width
        self._height = height

        self._module = Module(device.load_module("falcor2.utils"))
        self._reset_func = self._module.accumulator_reset
        self._output_func = self._module.accumulator_output
        self._update_func = self._module.accumulator_update
        self._update_and_output_func = self._module.accumulator_update_and_output
        self._update_and_output_inplace_func = self._module.accumulator_update_and_output_inplace

        self._accumulator = Tensor.empty(device, shape=(height, width), dtype=float4)

    def reset(self, cmd: Optional[CommandEncoder] = None):
        self._reset_func(self._accumulator, _append_to=cmd)

    def output(self, output: Tensor, cmd: Optional[CommandEncoder] = None):
        self._output_func(self._accumulator, output, _append_to=cmd)

    def update(self, input: Tensor, cmd: Optional[CommandEncoder] = None):
        self._update_func(input, self._accumulator, _append_to=cmd)

    def update_and_output(
        self, input: Tensor, output: Tensor, cmd: Optional[CommandEncoder] = None
    ):
        if input == output:
            self._update_and_output_inplace_func(input, self._accumulator, _append_to=cmd)
        else:
            self._update_and_output_func(input, self._accumulator, output, _append_to=cmd)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
