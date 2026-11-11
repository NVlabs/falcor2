# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any, Optional

import slangpy as spy
import falcor2 as fc


class SceneShaderHelper:
    """Resolve consumer requirements and cache a Slang module linked for the scene."""

    def __init__(self, device: spy.Device):
        self._device = device
        self._utils = spy.Module(device.load_module("falcor2.utils"))
        self._module: Optional[spy.Module] = None
        self._module_key: Optional[Any] = None
        self._scene = None
        self._scene_requirements: fc.SceneRequirements | None = None
        self._requirements_key = None

    def prepare(
        self,
        scene: fc.Scene,
        module: str | spy.Module,
        *,
        policy: fc.SceneCapabilityPolicy = fc.SceneCapabilityPolicy.all,
        specified_capabilities: fc.SceneCapabilities = fc.SceneCapabilities.all(),
        material_aovs: Sequence[str] = (),
        required_capabilities: fc.SceneCapabilities = fc.SceneCapabilities(),
    ) -> tuple[spy.Module, fc.SceneRequirements]:
        """Return the linked module and the exact requirements snapshot used to compose it.

        Call after scene.update(). Selection defaults to all capabilities. AOV names and
        required_capabilities describe consumer demand independently of output storage.
        specified_capabilities is used only with SceneCapabilityPolicy.specified.
        Use the returned snapshot for type conformances and ray-tracing setup. Unchanged
        scene generations and demand reuse resolution; failed preparation preserves the
        previous module, requirements, and bound scene.
        """
        if scene.device is not self._device:
            raise ValueError("Scene and shader helper must use the same device.")
        requested_aovs = tuple(material_aovs)
        # Generations are unique across scenes, including before their first update.
        requirements_key = (
            scene.requirements_generation,
            policy,
            specified_capabilities,
            requested_aovs,
            required_capabilities,
        )
        requirements = self._scene_requirements
        if requirements_key != self._requirements_key:
            requirements = scene.resolve_requirements(
                fc.SceneRenderRequest(
                    material_aovs=requested_aovs, required_capabilities=required_capabilities
                ),
                policy,
                specified_capabilities,
            )
        assert requirements is not None
        module_key = module
        if (
            self._module is None
            or module_key != self._module_key
            or not requirements.matches_shader_inputs(self._scene_requirements)
        ):
            render_module = spy.Module(scene.render_module)
            source_module = (
                self._device.load_module(module)
                if isinstance(module, str)
                else module.device_module
            )
            self._module = spy.Module(
                source_module,
                link=[self._utils, render_module] + [spy.Module(m) for m in requirements.modules],
            )
        self._module_key = module_key
        self._requirements_key = requirements_key
        self._scene_requirements = requirements
        self._scene = scene
        return self._module, requirements

    def bind_scene(self, cursor: spy.ShaderCursor) -> None:
        self._scene.bind(cursor)
