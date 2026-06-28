# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.rendernodes import PathTracerPipeline
from falcor2.testing.image_test_plugin import ImageTest


PROJECT_ROOT = Path(__file__).parents[3]
SCENE_MODULE_DIR = PROJECT_ROOT / "data/assets/test_normalmap"
WIDTH = 512
HEIGHT = 288
SPP = 16


@dataclass(frozen=True)
class NormalMapSceneConfig:
    id: str
    module_name: str


NORMALMAP_SCENES = [
    NormalMapSceneConfig("glb_bottom", "glb_bottom"),
    NormalMapSceneConfig("glb_right", "glb_right"),
    NormalMapSceneConfig("usd_bottom", "usd_bottom"),
    NormalMapSceneConfig("usd_right", "usd_right"),
]


def _load_scene_module(module_name: str) -> ModuleType:
    module_path = SCENE_MODULE_DIR / f"{module_name}.py"
    if not module_path.exists():
        pytest.skip(f"Normal map scene module is not available: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load scene module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    # Dynamically loaded scene modules need their directory on sys.path for sibling imports.
    scene_module_dir = str(SCENE_MODULE_DIR)
    inserted_scene_module_dir = False
    if scene_module_dir not in sys.path:
        sys.path.insert(0, scene_module_dir)
        inserted_scene_module_dir = True
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if inserted_scene_module_dir:
            sys.path.remove(scene_module_dir)
    return module


@pytest.mark.parametrize("device_type", [spy.DeviceType.d3d12])
@pytest.mark.parametrize("config", NORMALMAP_SCENES, ids=[config.id for config in NORMALMAP_SCENES])
def test_blender_sun_normalmap_scene(
    device_type: spy.DeviceType,
    config: NormalMapSceneConfig,
    image_test: ImageTest,
):
    if device_type not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip(f"{device_type} is not available in this test environment")
    device = helpers.get_device(device_type)
    scene_module = _load_scene_module(config.module_name)
    scene_path = getattr(scene_module, "SCENE_PATH", None)
    if scene_path is not None and not Path(scene_path).exists():
        pytest.skip(f"Normal map scene asset is not available: {scene_path}")
    scene, camera, metadata = scene_module.create_scene(device, width=WIDTH, height=HEIGHT)
    assert metadata["label"] == config.id

    pipeline = PathTracerPipeline.create(device)
    pipeline.spp = SPP
    pipeline.tone_map = True
    pipeline.path_tracer.enable_nee = True
    pipeline.path_tracer.enable_mis = True
    pipeline.path_tracer.enable_analytic_lights = True
    pipeline.path_tracer.enable_environment_light = True
    pipeline.path_tracer.enable_emissive_triangles = False
    pipeline.path_tracer.env_map_as_background = False
    pipeline.path_tracer.background_color = spy.float3(0.0, 0.0, 0.0)
    pipeline.path_tracer.max_depth = 3
    pipeline.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba16_float, (HEIGHT, WIDTH))

    image = pipeline(scene, camera).to_numpy()
    assert image.shape == (HEIGHT, WIDTH, 4)
    assert np.isfinite(image).all()
    assert image[..., :3].max() > 0.0

    image_test(image, tolerance=0.002)
