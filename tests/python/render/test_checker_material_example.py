# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import falcor2 as f2
import numpy as np
import pytest
import slangpy as spy
from falcor2.editor import create_device


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pathtracer"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from checker_material import CheckerMaterial  # noqa: E402


def load_scene_sampling_module(device: spy.Device, scene: f2.Scene) -> tuple[spy.Module, object]:
    from falcor2.editor import SceneShaderHelper

    helper = SceneShaderHelper(device)
    module = helper.get_module(scene, "falcor2.tools.materials.scene_sampling_tools")
    return module, helper


def test_checker_material_example() -> None:
    device = create_device(enable_debug_layers=True)
    scene = f2.Scene.create(device)

    props = f2.Properties()
    props["color_a"] = spy.float3(0.92, 0.62, 0.24)
    props["color_b"] = spy.float3(0.10, 0.18, 0.35)
    props["scale"] = 8.0

    material = scene.create_material(CheckerMaterial, props)
    scene.update()

    # Coverage note: required_module registration and the fixed-UV Lambert checks below
    # are now also exercised through test_materials2.py. The unique part left here is
    # the post-creation property mutation check that verifies scene.update() picks up
    # a scale change and flips the checker cell selection.
    required_module = material.required_module()
    modules = cast(list[spy.SlangModule], scene.requirements.modules)
    assert required_module is not None
    assert any(module.name == required_module.name for module in modules)

    module, helper = load_scene_sampling_module(device, scene)
    eval_simple = (
        module["material_eval_simple<TinyUniformSampleGenerator>"]
        .as_func()
        .write(helper.bind_scene)
    )

    seed = spy.uint3(7, 0, 7)
    wi = spy.float3(0, 1, 0)
    wo = spy.float3(0, 1, 0)

    def eval_albedo(uvs: spy.float2) -> spy.float3:
        result = cast(spy.float3, eval_simple(seed, material, uvs, wi, wo))
        return result * np.pi

    assert eval_albedo(spy.float2(0.50, 0.50)) == pytest.approx(spy.float3(0.92, 0.62, 0.24))
    assert eval_albedo(spy.float2(0.55, 0.65)) == pytest.approx(spy.float3(0.10, 0.18, 0.35))

    material.scale = 4.0
    scene.update()
    assert eval_albedo(spy.float2(0.55, 0.65)) == pytest.approx(spy.float3(0.92, 0.62, 0.24))
