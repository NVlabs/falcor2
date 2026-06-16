# SPDX-License-Identifier: Apache-2.0

"""Test image testing with various kronos scenes."""

from os import PathLike
import pytest
import falcor2.minitracer.tools as mt
import slangpy as spy
from pathlib import Path
import math
import falcor2.testing.helpers as helpers
from falcor2.testing.image_test_plugin import ImageTest
from falcor2.minitracer.scene import Scene

KRONOS_SCENE_NAMES = [
    "AlphaBlendModeTest",
    "ChronographWatch",
    "CompareRoughness",
    "DamagedHelmet",
    "EmissiveStrengthTest",
    "MultiUVTest",
    "SpecGlossVsMetalRough",
    "TextureLinearInterpolationTest",
    "TextureSettingsTest",
    "TextureTransformTest",
    "TransmissionTest",
    "VertexColorTest",
]
KRONOS_SCENE_PATHS = [f"data/assets/kronos/{name}/glTF/{name}.gltf" for name in KRONOS_SCENE_NAMES]


def get_resolution_and_spp(device_type: spy.DeviceType):
    width = 1024
    height = 1024
    spp = 1024
    if device_type == spy.DeviceType.cuda:
        width = 256
        height = 256
    return width, height, spp


def scene_setup(scene_path: PathLike[str] | str, scene: Scene, scene_from_world: spy.float4x4):
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


def renderer_setup(device: spy.Device):
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


def create_kronos_scene(device: spy.Device):
    """Create a scene with kronos assets."""
    scene = Scene(device)

    x = 0
    y = 0
    for path in KRONOS_SCENE_PATHS:
        scene_setup(path, scene, spy.math.matrix_from_translation(spy.float3(x - 1.5, y - 1, 0)))
        x += 1
        if x >= 4:
            x = 0
            y += 1
    return scene


# TODO: Known images that render incorrectly on Vulkan:
# - SpecGlossVsMetalRough: Big black splodge
# - CompareRoughness: Apparent texture mapping on rear of spheres
# - ChronographWatch: Subtle uv issue on back of watch
# - TextureSettingsTest: Subtle uv issue
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_render_kronos_sample_assets(device_type: spy.DeviceType, image_test: ImageTest):
    """Builds a single scene containing a set of Kronos sample assets and render it
    from the front and back. Disables MIS/NEE, focussed on testing the renderer
    features such as texture sampling / uv channels etc."""

    width, height, spp = get_resolution_and_spp(device_type)

    device = helpers.get_device(device_type)
    scene = create_kronos_scene(device)
    renderer = renderer_setup(device)
    camera = scene.create_camera(width, height, 45)

    # Set camera position based on angle
    for angle in [0, 180, 320, 40]:
        radius = 5
        camera.transform.pos = spy.float3(
            radius * math.sin(math.radians(angle)), 0.2, radius * math.cos(math.radians(angle))
        )
        camera.transform.rot = spy.math.quat_from_look_at(
            -spy.math.normalize(camera.transform.pos), spy.float3(0, 1, 0)
        )
        image = renderer.render(scene, camera, spp=spp)
        image_test(image, tolerance=0.001, additional_name=f"angle_{angle}")

        if angle == 0:
            guides = renderer.render_guides(scene, camera)
            for name, guide in guides.items():
                image_test(
                    guide,
                    tolerance=0.001,
                    additional_name=f"guide_{name}",
                    debug_output_stratify=name == "depth",
                )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mis(device_type: spy.DeviceType, image_test: ImageTest):
    """Builds a single scene containing a set of Kronos sample assets and render
    it with MIS/NEE enabled and env map visible."""

    width, height, spp = get_resolution_and_spp(device_type)

    device = helpers.get_device(device_type)
    scene = create_kronos_scene(device)
    renderer = renderer_setup(device)
    renderer.enable_nee = True
    renderer.enable_mis = True
    renderer.env_map_as_background = True
    camera = scene.create_camera(width, height, 45)

    # Set camera position based on angle
    for angle in [0, 180]:
        radius = 5
        camera.transform.pos = spy.float3(
            radius * math.sin(math.radians(angle)), 0.2, radius * math.cos(math.radians(angle))
        )
        camera.transform.rot = spy.math.quat_from_look_at(
            -spy.math.normalize(camera.transform.pos), spy.float3(0, 1, 0)
        )
        image = renderer.render(scene, camera, spp=spp)
        image_test(image, tolerance=0.001, additional_name=f"angle_{angle}")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ray_tracing_pipeline(device_type: spy.DeviceType, image_test: ImageTest):
    """Renders a scene using the ray tracing pipeline."""

    width, height, spp = get_resolution_and_spp(device_type)

    device = helpers.get_device(device_type)
    scene = create_kronos_scene(device)
    renderer = renderer_setup(device)
    renderer.enable_nee = True
    renderer.enable_mis = True
    renderer.env_map_as_background = True
    renderer.use_raytracing_pipeline = True
    camera = scene.create_camera(width, height, 45)

    # Set camera position based on angle
    for angle in [0, 180]:
        radius = 5
        camera.transform.pos = spy.float3(
            radius * math.sin(math.radians(angle)), 0.2, radius * math.cos(math.radians(angle))
        )
        camera.transform.rot = spy.math.quat_from_look_at(
            -spy.math.normalize(camera.transform.pos), spy.float3(0, 1, 0)
        )
        image = renderer.render(scene, camera, spp=spp)
        image_test(image, tolerance=0.001, additional_name=f"angle_{angle}")


if __name__ == "__main__":
    pytest.main([__file__, "-vs", "--image-tests"])
