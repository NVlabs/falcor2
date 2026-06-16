# SPDX-License-Identifier: Apache-2.0

from os import PathLike
from pathlib import Path
import slangpy as spy
import pytest
import numpy as np
import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.minitracer.scene import Scene
import falcor2.minitracer.tools as mt


def add_to_scene(scene_path: PathLike[str] | str, scene: Scene, scene_from_world: spy.float4x4):
    """Set up the scene for testing."""
    # Load the scene
    scene_path = Path(scene_path)
    if not scene_path.exists():
        pytest.skip(f"Scene file not found: {scene_path}")

    mt.load_scene(
        scene.device,
        scene_path,
        rescale_to=0.85,
        append_to_scene=scene,
        scene_from_world=scene_from_world,
    )

    # Add environment map
    if scene.env_map is None:
        envmap = scene.create_env_map("data/assets/envmaps/aerodynamics_workshop_512.hdr")
        envmap.scaling_factor = spy.float3(1)


def setup_renderer(device: spy.Device):
    """Set up the device for rendering."""
    renderer = mt.create_renderer(device)

    # Configure renderer for consistent results
    renderer.enable_nee = False
    renderer.enable_mis = False
    renderer.enable_emissive_triangles = True
    renderer.enable_env_map = True
    renderer.env_map_as_background = False
    renderer.background_color = spy.float3(0.3, 0.3, 0.5)
    renderer.max_depth = 3
    renderer.tone_map = True
    return renderer


def create_scene_and_renderer(device: spy.Device, scene_path: PathLike[str] | str):
    """Create a scene and renderer for testing."""
    scene = Scene(device)
    scene_from_world = spy.float4x4.identity()
    add_to_scene(scene_path, scene, scene_from_world)
    renderer = setup_renderer(device)
    return scene, renderer
