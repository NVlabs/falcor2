# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from slangpy import CommandEncoder, Device, Module

from falcor2.rendergraph import (
    RenderNode,
    Container,
    ContainerSpec,
)
from falcor2.rendergraph.image_format import channel_count


class TonemapperNode(RenderNode):
    def __init__(self, device: Device):
        super().__init__()
        self._device = device
        self._module = Module(device.load_module("falcor2.utils"))
        self._output = None

    @classmethod
    def create(cls, device: Device) -> "TonemapperNode":
        return cls(device)

    def _get_output(self, input: Any):
        self._output = Container.create_temp_like(self._device, input, current=self._output)
        return self._output

    def forward(self, input: Any, cmd: Optional[CommandEncoder] = None):
        output = self._get_output(input)
        output_spec = ContainerSpec.from_container(output)
        runtime_input = Container.to_render_layout(input)
        runtime_output = Container.to_render_layout(output)
        channels = channel_count(output_spec.format)
        if channels == 3:
            self._module.tonemap_aces_film_f3(runtime_input, _result=runtime_output, _append_to=cmd)
        elif channels == 4:
            self._module.tonemap_aces_film_f4(runtime_input, _result=runtime_output, _append_to=cmd)
        else:
            self._module.tonemap_aces_film(runtime_input, _result=runtime_output, _append_to=cmd)
        return output
