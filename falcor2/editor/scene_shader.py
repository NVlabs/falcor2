# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import slangpy as spy
import falcor2 as fc


class SceneShaderHelper:
    """Load and cache a slang module linked with a scene's render module and requirements."""

    def __init__(self, device: spy.Device):
        self._device = device
        self._utils = spy.Module(device.load_module("falcor2.utils"))
        self._module: Optional[spy.Module] = None
        self._module_key: Optional[Any] = None
        self._requirements_generation: Optional[int] = None
        self._scene = None

    def get_module(self, scene: fc.Scene, module: str | spy.Module) -> spy.Module:
        requirements_generation = scene.requirements_generation
        module_key = module
        if (
            self._module is None
            or module_key != self._module_key
            or requirements_generation != self._requirements_generation
        ):
            reqs = scene.requirements
            render_module = spy.Module(scene.render_module)
            source_module = (
                self._device.load_module(module)
                if isinstance(module, str)
                else module.device_module
            )
            self._module = spy.Module(
                source_module,
                link=[self._utils, render_module] + [spy.Module(m) for m in reqs.modules],
            )
            self._requirements_generation = requirements_generation
            self._module_key = module_key
        self._scene = scene
        return self._module

    def bind_scene(self, cursor):
        self._scene.bind(cursor)
