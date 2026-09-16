# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers
import numpy as np
from falcor2.rendernodes import (
    ReferencePathTracerNode,
    SchedulingMode,
    VisibilityRayMode,
)
from falcor2.rendernodes.reference_pathtracer_node import WRITE_GUIDE_INTERFACE


DATA = Path(__file__).resolve().parents[3] / "data"
GUIDE_TEST_SCENE = "data/assets/kronos/TransmissionTest/glTF/TransmissionTest.gltf"

MATERIALX_NORMAL_DEPENDENT_OPACITY = """<materialx version="1.39" colorspace="lin_rec709">
  <normal name="normal" type="vector3">
    <input name="space" type="string" value="world" />
  </normal>
  <dotproduct name="normal_alignment" type="float">
    <input name="in1" type="vector3" nodename="normal" />
    <input name="in2" type="vector3" value="0.0, 0.0, -1.0" />
  </dotproduct>
  <ifgreater name="opacity" type="float">
    <input name="value1" type="float" nodename="normal_alignment" />
    <input name="value2" type="float" value="0.6" />
    <input name="in1" type="float" value="1.0" />
    <input name="in2" type="float" value="0.0" />
  </ifgreater>
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <uniform_edf name="emission" type="EDF">
    <input name="color" type="color3" value="1.0, 0.0, 0.0" />
  </uniform_edf>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
    <input name="edf" type="EDF" nodename="emission" />
    <input name="opacity" type="float" nodename="opacity" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>"""

MATERIALX_BACKFACE_DIFFUSE = """<materialx version="1.39" colorspace="lin_rec709">
  <gltf_pbr name="pbr" type="surfaceshader" nodedef="ND_gltf_pbr_surfaceshader">
    <input name="base_color" type="color3" value="0.8, 0.05, 0.02" />
    <input name="metallic" type="float" value="0.0" />
    <input name="roughness" type="float" value="0.2" />
    <input name="specular" type="float" value="0.0" />
  </gltf_pbr>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="pbr" />
  </surfacematerial>
</materialx>"""

MATERIALX_DIELECTRIC = """<materialx version="1.39" colorspace="lin_rec709">
  <dielectric_bsdf name="glass" type="BSDF">
    <input name="weight" type="float" value="1.0" />
    <input name="tint" type="color3" value="1.0, 1.0, 1.0" />
    <input name="ior" type="float" value="1.0" />
    <input name="roughness" type="vector2" value="0.0, 0.0" />
    <input name="scatter_mode" type="string" value="T" uniform="true" />
  </dielectric_bsdf>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="glass" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>"""


def _standard_props(values: dict[str, object]) -> f2.Properties:
    return f2.Properties(values)


def _create_mdl_opacity_material(
    scene: f2.Scene, name: str, *, opacity: float | None = None
) -> f2.Material:
    props = f2.Properties()
    props["mdl_library_path"] = str(DATA / "assets/test_mdl_opacity")
    props["mdl_material_name"] = f"opacity_test::{name}"
    props["mdl_class_compilation"] = True
    if opacity is not None:
        props["opacity"] = opacity
    return scene.create_material(f2.MDLMaterial, props)


def _create_materialx_opacity_material(scene: f2.Scene, opacity: float) -> f2.Material:
    props = f2.Properties()
    props["mtlx_basepath"] = str(DATA / "assets/test_mtlx_opacity")
    props["mtlx_path"] = "opacity_test.mtlx"
    props["mtlx_node_name"] = "M"
    props["mtlx_editable_params"] = "*"
    props["inputs:surface_opacity"] = opacity
    return scene.create_material(f2.MaterialXMaterial, props)


def _add_quad(
    scene: f2.Scene,
    z: float,
    normal_z: float,
    material: f2.Material,
    size: float = 1.4,
    center_x: float = 0.0,
    shading_normal: tuple[float, float, float] | None = None,
) -> None:
    geom = scene.create_geometry(f2.StaticMeshGeometry)
    h = size * 0.5
    positions = np.array(
        [
            [center_x - h, -h, z],
            [center_x + h, -h, z],
            [center_x - h, h, z],
            [center_x + h, h, z],
        ],
        dtype=np.float32,
    )
    normal = shading_normal if shading_normal is not None else (0.0, 0.0, normal_z)
    normals = np.tile(np.array(normal, dtype=np.float32), (4, 1))
    tangents = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (4, 1))
    handedness = np.ones((4,), dtype=np.float32)
    texcoords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    indices = (
        np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)
        if normal_z > 0.0
        else np.array([[0, 2, 1], [2, 3, 1]], dtype=np.uint32)
    )
    geom.set_mesh_data(
        positions=positions,
        sub_mesh_indices=[indices],
        normals=normals,
        tangents=tangents,
        handedness=handedness,
        texcoords=texcoords,
        name="quad",
    )
    entity = scene.create_entity()
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = geom
    instance.materials = [material]


def _add_box(
    scene: f2.Scene,
    z_min: float,
    z_max: float,
    material: f2.Material,
    size: float = 1.4,
    center_x: float = 0.0,
) -> None:
    geom = scene.create_geometry(f2.StaticMeshGeometry)
    h = size * 0.5
    faces = [
        (
            np.array([center_x - h, h, z_min]),
            np.array([size, 0.0, 0.0]),
            np.array([0.0, -size, 0.0]),
            np.array([0.0, 0.0, -1.0]),
        ),
        (
            np.array([center_x - h, -h, z_max]),
            np.array([size, 0.0, 0.0]),
            np.array([0.0, size, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        (
            np.array([center_x - h, -h, z_min]),
            np.array([0.0, 0.0, z_max - z_min]),
            np.array([0.0, size, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
        ),
        (
            np.array([center_x + h, -h, z_min]),
            np.array([0.0, size, 0.0]),
            np.array([0.0, 0.0, z_max - z_min]),
            np.array([1.0, 0.0, 0.0]),
        ),
        (
            np.array([center_x - h, -h, z_min]),
            np.array([size, 0.0, 0.0]),
            np.array([0.0, 0.0, z_max - z_min]),
            np.array([0.0, -1.0, 0.0]),
        ),
        (
            np.array([center_x - h, h, z_min]),
            np.array([0.0, 0.0, z_max - z_min]),
            np.array([size, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ),
    ]

    positions = []
    normals = []
    tangents = []
    texcoords = []
    indices = []
    for origin, u, v, normal in faces:
        base = len(positions)
        positions.extend([origin, origin + u, origin + v, origin + u + v])
        normals.extend([normal] * 4)
        tangents.extend([u / np.linalg.norm(u)] * 4)
        texcoords.extend([[0, 0], [1, 0], [0, 1], [1, 1]])
        indices.extend([[base, base + 1, base + 2], [base + 2, base + 1, base + 3]])

    geom.set_mesh_data(
        positions=np.asarray(positions, dtype=np.float32),
        sub_mesh_indices=[np.asarray(indices, dtype=np.uint32)],
        normals=np.asarray(normals, dtype=np.float32),
        tangents=np.asarray(tangents, dtype=np.float32),
        handedness=np.ones((len(positions),), dtype=np.float32),
        texcoords=np.asarray(texcoords, dtype=np.float32),
        name="box",
    )
    entity = scene.create_entity()
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = geom
    instance.materials = [material]


def _make_quad_camera(scene: f2.Scene, width: int = 8, height: int = 8) -> f2.Camera:
    return helpers.create_test_camera(
        scene,
        width=width,
        height=height,
        fov_y=35,
        position=spy.float3(0.0, 0.0, -2.0),
        rotation=spy.math.quat_from_look_at(spy.float3(0.0, 0.0, 1.0), spy.float3(0.0, 1.0, 0.0)),
    )


def _load_guide_test_scene(device: spy.Device) -> f2.Scene:
    scene = f2.Scene.load(device, GUIDE_TEST_SCENE)
    scene.update()
    scene.update()
    return scene


def _render_mean(
    device: spy.Device,
    scene: f2.Scene,
    camera: f2.Camera,
    *,
    iterations: int = 1,
    enable_interior_tracking: bool = True,
    enable_homogeneous_media: bool = True,
    enable_nested_interiors: bool = False,
    enable_nee: bool = False,
    enable_mis: bool = True,
    enable_russian_roulette: bool = True,
    use_background_color: bool = True,
    max_depth: int = 3,
    rr_depth: int = 3,
    visibility_ray_mode: VisibilityRayMode | None = None,
    scheduling_mode: SchedulingMode = SchedulingMode.simple,
    light_sampler: f2.LightSampler | None = None,
) -> np.ndarray:
    scene.update()
    node = ReferencePathTracerNode.create(device)
    if light_sampler is not None:
        node.light_sampler = light_sampler
    node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    node.enable_interior_tracking = enable_interior_tracking
    node.enable_homogeneous_media = enable_homogeneous_media
    node.enable_nested_interiors = enable_nested_interiors
    node.enable_nee = enable_nee
    node.enable_mis = enable_mis
    node.enable_russian_roulette = enable_russian_roulette
    node.max_depth = max_depth
    node.rr_depth = rr_depth
    node.scheduling_mode = scheduling_mode
    if visibility_ray_mode is not None:
        node.visibility_ray_mode = visibility_ray_mode
    node.use_background_color = use_background_color
    node.background_color = spy.float3(0.0)

    total = None
    for iteration in range(iterations):
        image, _ = node(
            scene,
            camera,
            iteration=iteration,
            subpixel_offset=spy.float2(0.0, 0.0),
            subpixel_random_jitter=0.0,
        )
        data = image.to_numpy()[..., :3]
        total = data if total is None else total + data
    assert total is not None
    return total / float(iterations)


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "layering_mode",
    (f2.MaterialXLayeringMode.bsdf_mix,),
)
def test_pathtracer_materialx_backface(
    device_type: spy.DeviceType,
    device: spy.Device,
    layering_mode: f2.MaterialXLayeringMode,
) -> None:
    if device_type == spy.DeviceType.cuda:
        pytest.skip("MaterialXMaterial is disabled on CUDA")

    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.MaterialXMaterial,
        f2.Properties(
            {
                "mtlx_buffer": MATERIALX_BACKFACE_DIFFUSE,
                "mtlx_node_name": "M",
                "mtlx_layering_mode": layering_mode,
            }
        ),
    )
    _add_quad(scene, z=0.0, normal_z=1.0, material=material)
    env_entity = scene.create_entity()
    env_map = env_entity.create_component(f2.EnvMapLight)
    env_map["env_map_path"] = "data/assets/envmaps/aerodynamics_workshop_512.hdr"

    camera = _make_quad_camera(scene, width=16, height=16)

    image = _render_mean(
        device,
        scene,
        camera,
        iterations=4,
        enable_nee=False,
        enable_mis=False,
        use_background_color=False,
        max_depth=2,
    )

    assert np.isfinite(image).all()
    mean = image.mean(axis=(0, 1))
    assert mean[0] > 1e-4
    assert mean[0] > mean[1] * 5.0


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "layering_mode",
    (f2.MaterialXLayeringMode.bsdf_mix,),
)
def test_pathtracer_materialx_transmissive_exit_backface(
    device_type: spy.DeviceType,
    device: spy.Device,
    layering_mode: f2.MaterialXLayeringMode,
) -> None:
    if device_type == spy.DeviceType.cuda:
        pytest.skip("MaterialXMaterial is disabled on CUDA")

    scene = f2.Scene.create(device)
    boundary = scene.create_material(
        f2.MaterialXMaterial,
        f2.Properties(
            {
                "mtlx_buffer": MATERIALX_DIELECTRIC,
                "mtlx_node_name": "M",
                "mtlx_layering_mode": layering_mode,
            }
        ),
    )
    emitter = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(0.0),
                "emissive_factor": spy.float3(1.0),
                "double_sided": True,
            }
        ),
    )
    _add_box(scene, z_min=0.0, z_max=1.0, material=boundary)
    _add_quad(scene, z=1.5, normal_z=-1.0, material=emitter)

    image = _render_mean(
        device,
        scene,
        _make_quad_camera(scene, width=1, height=1),
        enable_interior_tracking=True,
        max_depth=3,
    )

    assert np.isfinite(image).all()
    assert np.min(image[0, 0]) > 1e-4


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_render_with_geometry(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """ReferencePathTracerNode private dispatch produces non-zero output with a loaded scene."""
    pt = ReferencePathTracerNode.create(device)
    cam = helpers.create_test_camera(helmet_scene, width=64, height=64, fov_y=45)
    color = spy.Tensor.empty(device, (64, 64), spy.float4)
    pt._render(helmet_scene, cam, color, iteration=0)
    data = color.to_numpy()
    assert pt._render_func is not None
    assert data.max() > 0.0
    assert not np.isnan(data).any()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_unlit_material_returns_baked_color_and_terminates(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.UnlitMaterial,
        f2.Properties({"base_color_factor": spy.float3(1.5, 0.25, 2.0)}),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=material)
    camera = _make_quad_camera(scene)

    unlit_color = np.array([1.5, 0.25, 2.0])
    black_image = _render_mean(
        device,
        scene,
        camera,
        enable_nee=False,
        max_depth=3,
    )

    np.testing.assert_allclose(
        black_image[4, 4],
        unlit_color,
        rtol=1e-4,
        atol=1e-5,
    )

    light_entity = scene.create_entity()
    constant_light = light_entity.create_component(f2.ConstantLight)
    constant_light.radiance = spy.float3(10.0, 20.0, 40.0)
    bright_image = _render_mean(
        device,
        scene,
        camera,
        enable_nee=True,
        max_depth=3,
    )
    np.testing.assert_allclose(
        bright_image[4, 4],
        unlit_color,
        rtol=1e-4,
        atol=1e-5,
    )

    node = ReferencePathTracerNode.create(device)
    node.guide_output_specs = {
        "material_color": f2.ContainerSpec.auto(),
        "emission": f2.ContainerSpec.auto(),
    }
    beauty, guides = node(scene, camera, iteration=0)
    np.testing.assert_allclose(
        beauty.to_numpy()[4, 4, :3],
        unlit_color,
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        guides["material_color"].to_numpy()[4, 4, :3],
        np.clip(unlit_color, 0.0, 1.0),
    )
    np.testing.assert_allclose(
        guides["emission"].to_numpy()[4, 4, :3],
        unlit_color,
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_trace_ray_visibility_render(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """The explicit visibility ray type links and renders on every RT backend."""
    node = ReferencePathTracerNode.create(device)
    node.visibility_ray_mode = VisibilityRayMode.trace_ray
    node.enable_nee = True
    camera = helpers.create_test_camera(helmet_scene, width=32, height=32, fov_y=45)

    color, _ = node(helmet_scene, camera, subpixel_random_jitter=0.0)
    data = color.to_numpy()

    assert np.isfinite(data).all()
    assert data.max() > 0.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_visibility_modes_match_when_ray_query_is_supported(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    if not device.has_feature(spy.Feature.ray_query):
        pytest.skip("Ray queries are not supported by this device")

    camera = helpers.create_test_camera(helmet_scene, width=16, height=16, fov_y=45)
    images = []
    for mode in (VisibilityRayMode.ray_query, VisibilityRayMode.trace_ray):
        node = ReferencePathTracerNode.create(device)
        node.visibility_ray_mode = mode
        node.enable_nee = True
        color, _ = node(helmet_scene, camera, subpixel_random_jitter=0.0)
        images.append(color.to_numpy())

    np.testing.assert_allclose(images[0], images[1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_ser_scheduler_render_when_supported(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """The SER scheduler links and renders when the device exposes SER."""
    if not device.has_feature(spy.Feature.shader_execution_reordering):
        pytest.skip("SER is not supported by this device")

    camera = helpers.create_test_camera(helmet_scene, width=32, height=32, fov_y=45)
    images = []
    for mode in (SchedulingMode.simple, SchedulingMode.ser):
        node = ReferencePathTracerNode.create(device)
        node.scheduling_mode = mode
        color, _ = node(helmet_scene, camera, subpixel_random_jitter=0.0)
        images.append(color.to_numpy())

    assert np.isfinite(images[1]).all()
    assert images[1].max() > 0.0
    np.testing.assert_allclose(images[0], images[1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_double_sided_backfaces_render(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    def render_backface(double_sided: bool) -> np.ndarray:
        scene = f2.Scene.create(device)
        material = scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(0.0, 0.0, 0.0),
                    "emissive_factor": spy.float3(1.0, 0.7, 0.4),
                    "double_sided": double_sided,
                }
            ),
        )
        _add_quad(scene, z=0.0, normal_z=1.0, material=material)
        camera = _make_quad_camera(scene)
        return _render_mean(device, scene, camera, max_depth=1)

    single_sided = render_backface(False)
    double_sided = render_backface(True)

    assert np.isfinite(single_sided).all()
    assert np.isfinite(double_sided).all()
    assert single_sided.max() < 1e-4
    assert double_sided.max() > 0.1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_opacity_runtime_transitions(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    def make_texture(value: tuple[float, float, float, float]) -> spy.Texture:
        return device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=1,
            height=1,
            usage=spy.TextureUsage.shader_resource,
            data=np.array([value], dtype=np.float32),
        )

    scene = f2.Scene.create(device)
    front = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(0.0, 0.0, 0.0),
                "emissive_factor": spy.float3(1.0, 0.0, 0.0),
                "double_sided": True,
                "alpha_mode": f2.AlphaMode.mask,
            }
        ),
    )
    rear = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(0.0, 0.0, 0.0),
                "emissive_factor": spy.float3(0.0, 1.0, 0.0),
                "double_sided": True,
            }
        ),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=front)
    _add_quad(scene, z=0.2, normal_z=-1.0, material=rear)
    camera = _make_quad_camera(scene)

    node = ReferencePathTracerNode.create(device)
    node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    node.enable_nee = False
    node.enable_mis = False
    node.max_depth = 1
    node.background_color = spy.float3(0.0, 0.0, 0.0)

    def render(iteration: int = 0) -> np.ndarray:
        scene.update()
        image, _ = node(
            scene,
            camera,
            iteration=iteration,
            subpixel_offset=spy.float2(0.0, 0.0),
            subpixel_random_jitter=0.0,
        )
        return image.to_numpy()[..., :3]

    def render_center(iteration: int = 0) -> np.ndarray:
        return render(iteration)[4, 4]

    missing_texture = render_center()
    assert missing_texture[0] > 0.1
    assert missing_texture[0] > missing_texture[1] * 10.0

    front.alpha_factor = 0.0
    zero_factor = render_center()
    assert zero_factor[1] > 0.1
    assert zero_factor[1] > zero_factor[0] * 10.0

    front.alpha_factor = 1.0
    front.base_color_texture = make_texture((0.0, 0.0, 0.0, 0.0))
    transparent = render_center()
    assert transparent[1] > 0.1
    assert transparent[1] > transparent[0] * 10.0

    front.alpha_mode = f2.AlphaMode.opaque
    opaque = render_center()
    assert opaque[0] > 0.1
    assert opaque[0] > opaque[1] * 10.0

    front.alpha_mode = f2.AlphaMode.mask
    front.base_color_texture = make_texture((0.0, 0.0, 0.0, 1.0))
    alpha_channel = render_center()
    assert alpha_channel[0] > alpha_channel[1] * 10.0

    representable_cutoff = 128.0 / 255.0
    front.base_color_texture = make_texture((0.0, 0.0, 0.0, representable_cutoff))
    front.alpha_cutoff = representable_cutoff
    equality_passes = render_center()
    assert equality_passes[0] > equality_passes[1] * 10.0

    front.alpha_mode = f2.AlphaMode.blend
    front.base_color_texture = make_texture((0.0, 0.0, 0.0, 0.0))
    blend_zero = render_center()
    assert blend_zero[1] > 0.1
    assert blend_zero[1] > blend_zero[0] * 10.0

    front.base_color_texture = make_texture((0.0, 0.0, 0.0, 1.0))
    blend_one = render_center()
    assert blend_one[0] > 0.1
    assert blend_one[0] > blend_one[1] * 10.0

    front.base_color_texture = make_texture((0.0, 0.0, 0.0, 0.5))
    blended_samples = render()
    assert np.all(np.max(blended_samples[..., :2], axis=-1) > 0.1)
    front_fraction = np.mean(blended_samples[..., 0] > blended_samples[..., 1])
    assert 0.25 < front_fraction < 0.75

    if device.has_feature(spy.Feature.shader_execution_reordering):
        node.scheduling_mode = SchedulingMode.ser
        ser_blended_samples = render()
        np.testing.assert_allclose(
            ser_blended_samples,
            blended_samples,
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_opacity_visibility_modes(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    def make_alpha_texture(alpha: float) -> spy.Texture:
        return device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=1,
            height=1,
            usage=spy.TextureUsage.shader_resource,
            data=np.array([[0.0, 0.0, 0.0, alpha]], dtype=np.float32),
        )

    transparent_texture = make_alpha_texture(0.0)
    scene = f2.Scene.create(device)
    surface = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(1.0, 1.0, 1.0),
                "roughness_factor": 1.0,
                "double_sided": True,
            }
        ),
    )
    occluder = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(0.0, 0.0, 0.0),
                "double_sided": True,
                "alpha_mode": f2.AlphaMode.mask,
                "base_color_texture": transparent_texture,
            }
        ),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=surface)
    _add_quad(scene, z=-0.4, normal_z=-1.0, material=occluder, size=0.3, center_x=0.4)

    light_entity = scene.create_entity()
    light_transform = f2.Transform()
    light_transform.translation = spy.float3(0.8, 0.0, -0.8)
    light_entity.transform = light_transform
    light = light_entity.create_component(f2.PointLight)
    light.intensity = spy.float3(8.0, 8.0, 8.0)
    camera = _make_quad_camera(scene, width=16, height=16)

    def render(mode: VisibilityRayMode) -> np.ndarray:
        scene.update()
        node = ReferencePathTracerNode.create(device)
        node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
        node.visibility_ray_mode = mode
        node.enable_nee = True
        node.enable_mis = False
        node.max_depth = 2
        image, _ = node(
            scene,
            camera,
            iteration=0,
            subpixel_offset=spy.float2(0.0, 0.0),
            subpixel_random_jitter=0.0,
        )
        return image.to_numpy()[..., :3]

    trace_ray_transparent = render(VisibilityRayMode.trace_ray)
    assert trace_ray_transparent[8, 8].mean() > 0.1

    occluder.alpha_mode = f2.AlphaMode.opaque
    trace_ray_opaque = render(VisibilityRayMode.trace_ray)
    assert trace_ray_opaque[8, 8].mean() < trace_ray_transparent[8, 8].mean() * 0.1

    occluder.alpha_mode = f2.AlphaMode.blend
    occluder.base_color_texture = make_alpha_texture(0.0)
    trace_ray_blend_zero = render(VisibilityRayMode.trace_ray)
    np.testing.assert_allclose(
        trace_ray_blend_zero,
        trace_ray_transparent,
        rtol=1e-4,
        atol=1e-5,
    )

    occluder.base_color_texture = make_alpha_texture(1.0)
    trace_ray_blend_one = render(VisibilityRayMode.trace_ray)
    np.testing.assert_allclose(
        trace_ray_blend_one,
        trace_ray_opaque,
        rtol=1e-4,
        atol=1e-5,
    )

    if device.has_feature(spy.Feature.ray_query):
        occluder.alpha_mode = f2.AlphaMode.mask
        occluder.base_color_texture = transparent_texture
        ray_query_transparent = render(VisibilityRayMode.ray_query)
        np.testing.assert_allclose(
            ray_query_transparent,
            trace_ray_transparent,
            rtol=1e-4,
            atol=1e-5,
        )

        occluder.alpha_mode = f2.AlphaMode.blend
        occluder.base_color_texture = make_alpha_texture(0.0)
        ray_query_blend_zero = render(VisibilityRayMode.ray_query)
        np.testing.assert_allclose(
            ray_query_blend_zero,
            trace_ray_blend_zero,
            rtol=1e-4,
            atol=1e-5,
        )

        occluder.base_color_texture = make_alpha_texture(1.0)
        ray_query_blend_one = render(VisibilityRayMode.ray_query)
        np.testing.assert_allclose(
            ray_query_blend_one,
            trace_ray_blend_one,
            rtol=1e-4,
            atol=1e-5,
        )

        occluder.base_color_texture = make_alpha_texture(0.5)
        trace_ray_blend = render(VisibilityRayMode.trace_ray)
        ray_query_blend = render(VisibilityRayMode.ray_query)
        np.testing.assert_allclose(
            ray_query_blend,
            trace_ray_blend,
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_mdl_cutout_opacity_camera_traversal(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    front = _create_mdl_opacity_material(scene, "parameterized", opacity=0.0)
    rear = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(0.0, 0.0, 0.0),
                "emissive_factor": spy.float3(0.0, 1.0, 0.0),
                "double_sided": True,
            }
        ),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=front)
    _add_quad(scene, z=0.2, normal_z=-1.0, material=rear)

    camera = _make_quad_camera(scene, width=16, height=16)

    def render(opacity: float, iterations: int = 1) -> np.ndarray:
        front["opacity"] = opacity
        return _render_mean(device, scene, camera, iterations=iterations, max_depth=1)

    transparent = render(0.0)
    opaque = render(1.0)
    blended = render(0.5)

    assert transparent[8, 8, 1] > 0.1
    assert transparent[8, 8, 1] > transparent[8, 8, 0] * 10.0
    assert np.max(opaque[8, 8]) < 1e-4
    assert np.isfinite(blended).all()
    rear_fraction = np.mean(blended[..., 1] > 0.1)
    assert 0.2 < rear_fraction < 0.8


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_mdl_cutout_opacity_visibility_modes(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    surface = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(1.0, 1.0, 1.0),
                "roughness_factor": 1.0,
                "double_sided": True,
            }
        ),
    )
    occluder = _create_mdl_opacity_material(scene, "parameterized", opacity=0.0)
    _add_quad(scene, z=0.0, normal_z=-1.0, material=surface)
    _add_quad(scene, z=-0.4, normal_z=-1.0, material=occluder, size=0.3, center_x=0.4)

    light_entity = scene.create_entity()
    light_transform = f2.Transform()
    light_transform.translation = spy.float3(0.8, 0.0, -0.8)
    light_entity.transform = light_transform
    light = light_entity.create_component(f2.PointLight)
    light.intensity = spy.float3(8.0, 8.0, 8.0)
    camera = _make_quad_camera(scene, width=16, height=16)

    def render(mode: VisibilityRayMode, opacity: float) -> np.ndarray:
        occluder["opacity"] = opacity
        scene.update()
        node = ReferencePathTracerNode.create(device)
        node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
        node.visibility_ray_mode = mode
        node.enable_nee = True
        node.enable_mis = False
        node.max_depth = 2
        image, _ = node(
            scene,
            camera,
            iteration=0,
            subpixel_offset=spy.float2(0.0, 0.0),
            subpixel_random_jitter=0.0,
        )
        return image.to_numpy()[..., :3]

    trace_transparent = render(VisibilityRayMode.trace_ray, 0.0)
    trace_opaque = render(VisibilityRayMode.trace_ray, 1.0)
    assert trace_transparent[8, 8].mean() > 0.1
    assert trace_opaque[8, 8].mean() < trace_transparent[8, 8].mean() * 0.1

    if device.has_feature(spy.Feature.ray_query):
        ray_query_transparent = render(VisibilityRayMode.ray_query, 0.0)
        ray_query_opaque = render(VisibilityRayMode.ray_query, 1.0)
        np.testing.assert_allclose(ray_query_transparent, trace_transparent, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(ray_query_opaque, trace_opaque, rtol=1e-4, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_materialx_cutout_opacity_camera_traversal(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    if device_type == spy.DeviceType.cuda:
        pytest.skip("MaterialXMaterial is disabled on CUDA")

    scene = f2.Scene.create(device)
    front = _create_materialx_opacity_material(scene, 0.0)
    rear = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(0.0, 0.0, 0.0),
                "emissive_factor": spy.float3(0.0, 1.0, 0.0),
                "double_sided": True,
            }
        ),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=front)
    _add_quad(scene, z=0.2, normal_z=-1.0, material=rear)

    camera = _make_quad_camera(scene, width=16, height=16)

    def render(opacity: float) -> np.ndarray:
        front["inputs:surface_opacity"] = opacity
        return _render_mean(device, scene, camera, max_depth=1)

    transparent = render(0.0)
    opaque = render(1.0)
    blended = render(0.5)

    assert transparent[8, 8, 1] > 0.1
    assert transparent[8, 8, 1] > transparent[8, 8, 0] * 10.0
    assert opaque[8, 8, 0] > 0.1
    assert opaque[8, 8, 0] > opaque[8, 8, 1] * 10.0
    assert np.isfinite(blended).all()
    rear_fraction = np.mean(blended[..., 1] > 0.1)
    front_fraction = np.mean(blended[..., 0] > 0.1)
    assert 0.2 < rear_fraction < 0.8
    assert 0.2 < front_fraction < 0.8
    accepted_front = blended[..., 0][blended[..., 0] > 0.1]
    assert np.mean(accepted_front) == pytest.approx(np.mean(opaque[..., 0]), rel=0.05)

    for layering_mode in (
        f2.MaterialXLayeringMode.closure_tree,
        f2.MaterialXLayeringMode.bsdf_mix,
    ):
        normal_scene = f2.Scene.create(device)
        props = f2.Properties(
            {
                "mtlx_buffer": MATERIALX_NORMAL_DEPENDENT_OPACITY,
                "mtlx_node_name": "M",
                "mtlx_layering_mode": layering_mode,
            }
        )
        normal_front = normal_scene.create_material(f2.MaterialXMaterial, props)
        normal_rear = normal_scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(0.0, 0.0, 0.0),
                    "emissive_factor": spy.float3(0.0, 1.0, 0.0),
                    "double_sided": True,
                }
            ),
        )
        _add_quad(
            normal_scene,
            z=0.0,
            normal_z=-1.0,
            material=normal_front,
            # Exercise both the public and internal shading-normal repair policies.
            shading_normal=(0.0, 0.99995, -0.01),
        )
        _add_quad(normal_scene, z=0.2, normal_z=-1.0, material=normal_rear)
        normal_camera = _make_quad_camera(normal_scene, width=16, height=16)

        normal_image = _render_mean(device, normal_scene, normal_camera, max_depth=1)
        center = normal_image[8, 8]
        assert center[0] > 0.1
        assert center[0] > center[1] * 10.0


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_materialx_cutout_opacity_visibility_modes(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    if device_type == spy.DeviceType.cuda:
        pytest.skip("MaterialXMaterial is disabled on CUDA")

    scene = f2.Scene.create(device)
    surface = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(1.0, 1.0, 1.0),
                "roughness_factor": 1.0,
                "double_sided": True,
            }
        ),
    )
    occluder = _create_materialx_opacity_material(scene, 0.0)
    _add_quad(scene, z=0.0, normal_z=-1.0, material=surface)
    _add_quad(scene, z=-0.4, normal_z=-1.0, material=occluder, size=0.3, center_x=0.4)

    light_entity = scene.create_entity()
    light_transform = f2.Transform()
    light_transform.translation = spy.float3(0.8, 0.0, -0.8)
    light_entity.transform = light_transform
    light = light_entity.create_component(f2.PointLight)
    light.intensity = spy.float3(8.0, 8.0, 8.0)
    camera = _make_quad_camera(scene, width=16, height=16)

    def render(mode: VisibilityRayMode, opacity: float) -> np.ndarray:
        occluder["inputs:surface_opacity"] = opacity
        return _render_mean(
            device,
            scene,
            camera,
            enable_nee=True,
            enable_mis=False,
            max_depth=2,
            visibility_ray_mode=mode,
        )

    trace_transparent = render(VisibilityRayMode.trace_ray, 0.0)
    trace_opaque = render(VisibilityRayMode.trace_ray, 1.0)
    assert trace_transparent[8, 8].mean() > 0.1
    assert trace_opaque[8, 8].mean() < trace_transparent[8, 8].mean() * 0.1

    if device.has_feature(spy.Feature.ray_query):
        ray_query_transparent = render(VisibilityRayMode.ray_query, 0.0)
        ray_query_opaque = render(VisibilityRayMode.ray_query, 1.0)
        np.testing.assert_allclose(ray_query_transparent, trace_transparent, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(ray_query_opaque, trace_opaque, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_transmission_nee_sees_lower_hemisphere_light(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    def render_transmission(
        diffuse_transmission: float,
        light_sampler: f2.LightSampler | None = None,
    ) -> np.ndarray:
        scene = f2.Scene.create(device)
        surface = scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(1.0, 1.0, 1.0),
                    "roughness_factor": 1.0,
                    "transmission_factor": spy.float3(1.0, 1.0, 1.0),
                    "diffuse_transmission_factor": diffuse_transmission,
                    "thin_walled": True,
                    "double_sided": True,
                }
            ),
        )
        emitter = scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(0.0, 0.0, 0.0),
                    "emissive_factor": spy.float3(4.0, 4.0, 4.0),
                    "double_sided": True,
                }
            ),
        )
        _add_quad(scene, z=0.0, normal_z=-1.0, material=surface)
        _add_quad(scene, z=0.8, normal_z=-1.0, material=emitter)
        camera = _make_quad_camera(scene)
        return _render_mean(
            device,
            scene,
            camera,
            iterations=4,
            enable_nee=True,
            enable_mis=False,
            max_depth=2,
            light_sampler=light_sampler,
        )

    opaque = render_transmission(0.0)
    transmitted = render_transmission(1.0)
    hierarchical = f2.HierarchicalLightSampler()
    hierarchical.max_triangle_count_per_leaf = 1
    hierarchical_transmitted = render_transmission(1.0, hierarchical)

    assert np.isfinite(opaque).all()
    assert np.isfinite(transmitted).all()
    assert np.isfinite(hierarchical_transmitted).all()
    assert transmitted.mean() > opaque.mean() + 1e-3
    assert hierarchical_transmitted.mean() > opaque.mean() + 1e-3
    np.testing.assert_allclose(hierarchical_transmitted, transmitted, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_homogeneous_boundary_nee_uses_vacuum_side(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    def make_boundary(scene: f2.Scene) -> f2.Material:
        return scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(1.0),
                    "roughness_factor": 0.5,
                    "ior": 1.0,
                    "transmission_factor": spy.float3(1.0),
                    "diffuse_transmission_factor": 1.0,
                }
            ),
        )

    def add_point_light(scene: f2.Scene, z: float) -> None:
        light_entity = scene.create_entity()
        light_transform = f2.Transform()
        light_transform.translation = spy.float3(0.0, 0.0, z)
        light_entity.transform = light_transform
        light = light_entity.create_component(f2.PointLight)
        light.intensity = spy.float3(8.0)

    def render_entry(enable_homogeneous_media: bool) -> np.ndarray:
        scene = f2.Scene.create(device)
        boundary = make_boundary(scene)
        _add_quad(scene, z=0.0, normal_z=-1.0, material=boundary, size=4.0)
        add_point_light(scene, 0.25)
        return _render_mean(
            device,
            scene,
            _make_quad_camera(scene, width=1, height=1),
            iterations=8,
            enable_homogeneous_media=enable_homogeneous_media,
            enable_nee=True,
            enable_mis=False,
            max_depth=2,
        )

    def render_exit() -> np.ndarray:
        scene = f2.Scene.create(device)
        boundary = make_boundary(scene)
        _add_quad(scene, z=0.0, normal_z=-1.0, material=boundary, size=4.0)
        _add_quad(scene, z=1.0, normal_z=1.0, material=boundary, size=4.0)
        add_point_light(scene, 1.25)
        return _render_mean(
            device,
            scene,
            _make_quad_camera(scene, width=1, height=1),
            iterations=512,
            enable_nee=True,
            enable_mis=False,
            max_depth=3,
        )

    medium_entry = render_entry(True)
    clear_entry = render_entry(False)
    exit_with_nee = render_exit()

    np.testing.assert_allclose(medium_entry, 0.0, rtol=0.0, atol=1e-7)
    assert np.min(clear_entry[0, 0]) > 1e-4
    assert np.min(exit_with_nee[0, 0]) > 1e-4


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_homogeneous_extinction_matches_beer_lambert(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    extinction = np.log(np.array([2.0, 4.0, 8.0], dtype=np.float32))

    def render(
        sigma_a: np.ndarray, sigma_s: np.ndarray, enable_homogeneous_media: bool = True
    ) -> np.ndarray:
        scene = f2.Scene.create(device)
        medium = scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(1.0),
                    "roughness_factor": 0.0,
                    "ior": 1.0,
                    "transmission_factor": spy.float3(1.0),
                    "specular_transmission_factor": 1.0,
                    "volume_sigma_a": spy.float3(*sigma_a),
                    "volume_sigma_s": spy.float3(*sigma_s),
                }
            ),
        )
        emitter = scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(0.0),
                    "emissive_factor": spy.float3(1.0),
                    "double_sided": True,
                }
            ),
        )
        _add_box(scene, z_min=0.0, z_max=1.0, material=medium)
        _add_quad(scene, z=1.5, normal_z=-1.0, material=emitter)
        return _render_mean(
            device,
            scene,
            _make_quad_camera(scene, width=1, height=1),
            enable_homogeneous_media=enable_homogeneous_media,
            max_depth=3,
        )

    vacuum = render(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    attenuated = render(extinction, np.zeros(3, dtype=np.float32))
    media_disabled = render(extinction, np.zeros(3, dtype=np.float32), False)

    assert np.isfinite(vacuum).all()
    assert np.isfinite(attenuated).all()
    assert np.isfinite(media_disabled).all()
    assert np.min(vacuum[0, 0]) > 1e-4
    np.testing.assert_allclose(media_disabled, vacuum, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        attenuated[0, 0] / vacuum[0, 0],
        np.exp(-extinction),
        rtol=3e-3,
        atol=3e-4,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_homogeneous_forward_scattering_preserves_transmission(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    def render(
        sigma_a: float,
        sigma_s: float,
        anisotropy: float,
        scheduling_mode: SchedulingMode = SchedulingMode.simple,
    ) -> np.ndarray:
        scene = f2.Scene.create(device)
        medium = scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(1.0),
                    "roughness_factor": 0.0,
                    "ior": 1.0,
                    "transmission_factor": spy.float3(1.0),
                    "specular_transmission_factor": 1.0,
                    "volume_sigma_a": spy.float3(sigma_a),
                    "volume_sigma_s": spy.float3(sigma_s),
                    "volume_anisotropy": anisotropy,
                }
            ),
        )
        emitter = scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(0.0),
                    "emissive_factor": spy.float3(1.0),
                    "double_sided": True,
                }
            ),
        )
        _add_box(scene, z_min=0.0, z_max=1.0, material=medium)
        _add_quad(scene, z=1.5, normal_z=-1.0, material=emitter)
        return _render_mean(
            device,
            scene,
            _make_quad_camera(scene, width=1, height=1),
            iterations=512,
            max_depth=8,
            scheduling_mode=scheduling_mode,
        )

    vacuum = render(0.0, 0.0, 0.0)
    forward_scattered = render(0.0, 0.5, 0.999)

    assert np.isfinite(forward_scattered).all()
    assert np.min(vacuum[0, 0]) > 1e-4
    assert np.min(forward_scattered[0, 0] / vacuum[0, 0]) > 0.9

    if device.has_feature(spy.Feature.shader_execution_reordering):
        reordered = render(0.0, 0.5, 0.999, SchedulingMode.ser)
        np.testing.assert_allclose(reordered, forward_scattered, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_disables_nee_inside_tracked_interior(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    def render(thin_walled: bool) -> np.ndarray:
        scene = f2.Scene.create(device)
        boundary = scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(1.0),
                    "roughness_factor": 0.0,
                    "ior": 1.0,
                    "transmission_factor": spy.float3(1.0),
                    "specular_transmission_factor": 1.0,
                    "thin_walled": thin_walled,
                }
            ),
        )
        receiver = scene.create_material(
            f2.StandardMaterial,
            _standard_props(
                {
                    "base_color_factor": spy.float3(1.0),
                    "roughness_factor": 1.0,
                    "double_sided": True,
                }
            ),
        )
        _add_quad(scene, z=0.0, normal_z=-1.0, material=boundary)
        _add_quad(scene, z=0.4, normal_z=-1.0, material=receiver)

        light_entity = scene.create_entity()
        light_transform = f2.Transform()
        light_transform.translation = spy.float3(0.0, 0.0, 0.2)
        light_entity.transform = light_transform
        light = light_entity.create_component(f2.PointLight)
        light.intensity = spy.float3(8.0)

        return _render_mean(
            device,
            scene,
            _make_quad_camera(scene, width=1, height=1),
            enable_nee=True,
            enable_mis=False,
            max_depth=3,
        )

    inside_solid = render(False)
    thin_walled = render(True)

    np.testing.assert_allclose(inside_solid, 0.0, rtol=0.0, atol=1e-7)
    assert np.min(thin_walled[0, 0]) > 1e-4


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_environment_direct_hit_mis_is_finite(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(1.0, 1.0, 1.0),
                "roughness_factor": 1.0,
                "double_sided": True,
            }
        ),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=material)
    env_entity = scene.create_entity()
    env_map = env_entity.create_component(f2.EnvMapLight)
    env_map["env_map_path"] = "data/assets/envmaps/aerodynamics_workshop_512.hdr"

    camera = _make_quad_camera(scene)
    image = _render_mean(
        device,
        scene,
        camera,
        iterations=8,
        enable_nee=True,
        max_depth=2,
    )

    assert np.isfinite(image).all()
    assert image.mean() > 1e-4


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_small_distant_light_nee_converges(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(1.0, 1.0, 1.0),
                "roughness_factor": 1.0,
                "double_sided": True,
            }
        ),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=material)

    light_entity = scene.create_entity()
    light_transform = f2.Transform()
    light_transform.rotation = spy.quatf(1.0, 0.0, 0.0, 0.0)
    light_entity.transform = light_transform
    distant_light = light_entity.create_component(f2.DistantLight)
    distant_light.radiance = spy.float3(10000.0)
    distant_light.cutoff_angle = 0.26785

    image = _render_mean(
        device,
        scene,
        _make_quad_camera(scene),
        iterations=1,
        enable_nee=True,
        enable_mis=True,
        max_depth=2,
    )

    assert np.isfinite(image).all()
    assert image[4, 4].mean() > 0.01


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_environment_intensity_and_exposure_scale_background(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    env_entity = scene.create_entity()
    env_map = env_entity.create_component(f2.EnvMapLight)
    env_map.env_map_path = "data/assets/envmaps/aerodynamics_workshop_512.hdr"
    camera = _make_quad_camera(scene)

    baseline = _render_mean(
        device,
        scene,
        camera,
        use_background_color=False,
        max_depth=1,
    )

    env_map.intensity = spy.float3(0.25, 0.5, 0.75)
    env_map.exposure = 1.0
    scaled = _render_mean(
        device,
        scene,
        camera,
        use_background_color=False,
        max_depth=1,
    )

    expected_scale = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    np.testing.assert_allclose(scaled, baseline * expected_scale, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_multiple_environment_lights_sum_background(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    expected = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    scene = f2.Scene.create(device)

    distant_entity = scene.create_entity()
    distant_light = distant_entity.create_component(f2.DistantLight)
    distant_light.radiance = spy.float3(0.25, 0.0, 0.0)
    distant_light.cutoff_angle = 180.0

    constant_entity = scene.create_entity()
    constant_light = constant_entity.create_component(f2.ConstantLight)
    constant_light.radiance = spy.float3(0.0, 0.5, 0.75)

    camera = _make_quad_camera(scene)
    image = _render_mean(
        device,
        scene,
        camera,
        use_background_color=False,
        max_depth=1,
    )

    np.testing.assert_allclose(image, np.broadcast_to(expected, image.shape), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_multiple_environment_lights_nee_mis_matches_order_and_reference(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(1.0, 1.0, 1.0),
                "roughness_factor": 1.0,
                "double_sided": True,
            }
        ),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=material)

    lights = []
    for radiance in (spy.float3(1.0, 0.0, 0.0), spy.float3(0.0, 1.0, 0.0)):
        entity = scene.create_entity()
        light = entity.create_component(f2.ConstantLight)
        light.radiance = radiance
        lights.append(light)

    camera = _make_quad_camera(scene)

    def render() -> np.ndarray:
        return _render_mean(
            device,
            scene,
            camera,
            iterations=8,
            enable_nee=True,
            enable_mis=True,
            max_depth=2,
        )

    combined = render()
    lights[0].radiance = spy.float3(0.0, 1.0, 0.0)
    lights[1].radiance = spy.float3(1.0, 0.0, 0.0)
    reordered = render()
    lights[0].radiance = spy.float3(1.0, 1.0, 0.0)
    lights[1].active = False
    reference = render()

    assert np.isfinite(combined).all()
    assert np.isfinite(reordered).all()
    assert np.isfinite(reference).all()
    np.testing.assert_allclose(combined, reordered, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(combined[..., 0], combined[..., 1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        combined.mean(axis=(0, 1)),
        reference.mean(axis=(0, 1)),
        rtol=0.2,
        atol=0.02,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_delta_reflection_environment_with_nee_without_mis(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(1.0, 1.0, 1.0),
                "metallic_factor": 1.0,
                "roughness_factor": 0.0,
                "double_sided": True,
            }
        ),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=material)
    env_entity = scene.create_entity()
    env_map = env_entity.create_component(f2.EnvMapLight)
    env_map["env_map_path"] = "data/assets/envmaps/aerodynamics_workshop_512.hdr"

    camera = _make_quad_camera(scene)
    image = _render_mean(
        device,
        scene,
        camera,
        enable_nee=True,
        enable_mis=False,
        max_depth=2,
    )

    assert np.isfinite(image).all()
    assert image.mean() > 1e-4


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_settings_change(device_type: spy.DeviceType, device: spy.Device) -> None:
    """Changing constants and settings updates their respective dictionaries."""
    pt = ReferencePathTracerNode.create(device)
    assert pt.enable_interior_tracking is True
    assert pt.enable_homogeneous_media is True
    assert pt.enable_nested_interiors is False
    assert pt.enable_russian_roulette is True
    assert pt._constants["ENABLE_INTERIOR_TRACKING"] is True
    assert pt._constants["ENABLE_HOMOGENEOUS_MEDIA"] is True
    assert pt._constants["ENABLE_NESTED_INTERIORS"] is False
    assert pt._constants["ENABLE_RUSSIAN_ROULETTE"] is True
    pt.enable_interior_tracking = False
    pt.enable_homogeneous_media = False
    pt.enable_nested_interiors = True
    pt.enable_russian_roulette = False
    assert pt._constants["ENABLE_INTERIOR_TRACKING"] is False
    assert pt._constants["ENABLE_HOMOGENEOUS_MEDIA"] is False
    assert pt._constants["ENABLE_NESTED_INTERIORS"] is True
    assert pt._constants["ENABLE_RUSSIAN_ROULETTE"] is False
    pt.enable_nee = True
    assert pt._constants["ENABLE_NEE"] is True
    assert pt._settings["max_depth"] == 3
    pt.max_depth = 5
    assert pt._settings["max_depth"] == 5
    assert pt.rr_depth == 3
    assert pt._settings["rr_depth"] == 3
    pt.rr_depth = 0
    assert pt.rr_depth == 1
    pt.rr_depth = 256
    assert pt.rr_depth == 255
    pt.rr_depth = 4
    assert pt._settings["rr_depth"] == 4
    assert pt.use_background_color is False
    assert pt._settings["use_background_color"] is False
    pt.use_background_color = True
    pt.background_color = spy.float3(0.1, 0.2, 0.3)
    assert pt._settings["use_background_color"] is True
    assert pt._settings["background_color"] == spy.float3(0.1, 0.2, 0.3)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_russian_roulette_is_unbiased(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Russian roulette changes individual paths while preserving their expected contribution."""
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.StandardMaterial,
        _standard_props(
            {
                "base_color_factor": spy.float3(0.25, 0.25, 0.25),
                "roughness_factor": 1.0,
                "double_sided": True,
            }
        ),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=material, size=4.0)
    light_entity = scene.create_entity()
    constant_light = light_entity.create_component(f2.ConstantLight)
    constant_light.radiance = spy.float3(1.0, 1.0, 1.0)
    camera = _make_quad_camera(scene, width=128, height=128)

    reference = _render_mean(
        device,
        scene,
        camera,
        enable_russian_roulette=False,
        use_background_color=False,
        max_depth=2,
    )
    roulette = _render_mean(
        device,
        scene,
        camera,
        enable_russian_roulette=True,
        use_background_color=False,
        max_depth=2,
        rr_depth=1,
    )

    assert np.isfinite(roulette).all()
    assert not np.array_equal(roulette, reference)
    np.testing.assert_allclose(
        roulette.mean(axis=(0, 1)),
        reference.mean(axis=(0, 1)),
        rtol=0.08,
        atol=0.01,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_scheduler_and_visibility_modes(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    node = ReferencePathTracerNode.create(device)

    assert node.scheduling_mode == SchedulingMode.simple
    expected_visibility_mode = (
        VisibilityRayMode.ray_query
        if device.has_feature(spy.Feature.ray_query)
        else VisibilityRayMode.trace_ray
    )
    assert node.visibility_ray_mode == expected_visibility_mode
    assert node._constants["VISIBILITY_RAY_MODE"] == int(expected_visibility_mode)

    node.visibility_ray_mode = VisibilityRayMode.trace_ray
    assert node._constants["VISIBILITY_RAY_MODE"] == 1

    if device.has_feature(spy.Feature.ray_query):
        node.visibility_ray_mode = VisibilityRayMode.ray_query
        assert node._constants["VISIBILITY_RAY_MODE"] == 0
    else:
        with pytest.raises(RuntimeError, match="Ray-query visibility"):
            node.visibility_ray_mode = VisibilityRayMode.ray_query

    if device.has_feature(spy.Feature.shader_execution_reordering):
        node.scheduling_mode = SchedulingMode.ser
        assert node._constants["SCHEDULING_MODE"] == int(SchedulingMode.ser)
    else:
        with pytest.raises(RuntimeError, match="SER scheduling"):
            node.scheduling_mode = SchedulingMode.ser


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_module_caching(
    device_type: spy.DeviceType, device: spy.Device, empty_scene: f2.Scene
) -> None:
    """_get_module returns cached module when requirements unchanged."""
    pt = ReferencePathTracerNode.create(device)
    mod1 = pt._get_module(empty_scene)
    mod2 = pt._get_module(empty_scene)
    assert mod1 is mod2


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_path_tracer_respects_explicit_output_dims(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    camera = helpers.create_test_camera(helmet_scene, width=64, height=64)
    node = ReferencePathTracerNode.create(device)
    node.output_spec = f2.ContainerSpec.texture2d(dims=(24, 40))

    image, _ = node(helmet_scene, camera)

    assert image.height == 24
    assert image.width == 40
    assert camera.height == 64
    assert camera.width == 64


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_path_tracer_resets_previous_camera_when_render_dims_change(
    device_type: spy.DeviceType,
    device: spy.Device,
    empty_scene: f2.Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = helpers.create_test_camera(empty_scene, width=64, height=64)
    node = ReferencePathTracerNode.create(device)
    previous_cameras = []
    ray_samplers = []

    class FakeRenderFunc:
        def write(self, _bind):
            return self

        def call(self, **kwargs):
            previous_cameras.append(kwargs["previous_camera"])
            ray_samplers.append(kwargs["ray_sampler"])

    fake_func = FakeRenderFunc()
    monkeypatch.setattr(node, "_get_module", lambda _scene: object())
    monkeypatch.setattr(node, "_get_render_func", lambda _module, _write_guide: fake_func)

    node._render(empty_scene, camera, spy.Tensor.empty(device, (24, 32), spy.float4), 0)
    first_previous = previous_cameras[-1]
    assert first_previous.dims == spy.uint2(32, 24)
    assert isinstance(first_previous, f2.CameraUniforms)
    assert isinstance(ray_samplers[-1], f2.CameraUniforms)
    assert ray_samplers[-1].width == 32
    assert ray_samplers[-1].height == 24

    node._render(empty_scene, camera, spy.Tensor.empty(device, (12, 16), spy.float4), 1)

    assert previous_cameras[-1].dims == spy.uint2(16, 12)
    assert previous_cameras[-1] is not first_previous
    assert isinstance(previous_cameras[-1], f2.CameraUniforms)
    assert ray_samplers[-1].width == 16
    assert ray_samplers[-1].height == 12
    assert node._previous_camera_dims == (16, 12)
    assert camera.width == 64
    assert camera.height == 64


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_path_tracer_uses_camera_dims_as_fallback(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    camera = helpers.create_test_camera(helmet_scene, width=37, height=23)
    node = ReferencePathTracerNode.create(device)

    image, _ = node(helmet_scene, camera)

    assert image.height == 23
    assert image.width == 37


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_path_tracer_default_color_format_by_backend(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    node = ReferencePathTracerNode.create(device)
    expected = (
        spy.Format.rgba32_float
        if device.desc.type == spy.DeviceType.cuda
        else spy.Format.rgba16_float
    )

    assert node._default_color_format() == expected


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_non_dlss_guides_with_geometry(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """ReferencePathTracerNode renders non-DLSS guide outputs."""
    scene = _load_guide_test_scene(device)
    node = ReferencePathTracerNode.create(device)
    node.guide_output_specs = {
        "material_color": f2.ContainerSpec.auto(),
        "metallic": f2.ContainerSpec.auto(),
        "emission": f2.ContainerSpec.auto(),
        "geometry_id": f2.ContainerSpec.auto(),
    }
    camera_position = spy.float3(0.0, 0.0, 4.0)
    cam = helpers.create_test_camera(
        scene,
        width=16,
        height=12,
        fov_y=45,
        position=camera_position,
        rotation=spy.math.quat_from_look_at(
            -spy.math.normalize(camera_position),
            spy.float3(0.0, 1.0, 0.0),
        ),
    )

    _, guides = node(scene, cam, iteration=0)

    assert isinstance(guides["material_color"], spy.Texture)
    assert isinstance(guides["metallic"], spy.Texture)
    assert isinstance(guides["emission"], spy.Texture)
    assert isinstance(guides["geometry_id"], spy.Texture)
    for name, value in guides.items():
        if name not in {"material_color", "metallic", "emission", "geometry_id"}:
            assert value is None

    material_color = guides["material_color"].to_numpy()
    metallic = guides["metallic"].to_numpy()
    emission = guides["emission"].to_numpy()
    geometry_id = guides["geometry_id"].to_numpy()
    assert material_color.shape == (12, 16, 4)
    assert metallic.shape == (12, 16)
    assert emission.shape == (12, 16, 4)
    assert geometry_id.shape == (12, 16, 4)
    assert np.isfinite(material_color).all()
    assert np.isfinite(emission).all()
    assert material_color[..., :3].max() > 0.0
    assert metallic.min() >= 0.0
    assert metallic.max() <= 1.0
    assert metallic.max() > 0.0
    assert emission[..., :3].max() > 0.0
    valid_geometry = geometry_id[..., 1] != np.uint32(0xFFFFFFFF)
    assert valid_geometry.any()
    assert (geometry_id[..., 0][valid_geometry] == int(f2.GeometryType.triangle)).all()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    ("material_type", "properties", "expected_color", "expected_metallic"),
    [
        (
            f2.StandardMaterial,
            {
                "base_color_factor": spy.float3(0.25, 0.5, 0.75),
                "metallic_factor": 0.75,
            },
            np.array([0.25, 0.5, 0.75]),
            0.75,
        ),
        (
            "OpenPBRMaterial",
            {
                "base_color": spy.float3(0.5, 0.75, 1.0),
                "base_weight": 0.5,
                "base_metalness": 0.75,
            },
            np.array([0.25, 0.375, 0.5]),
            0.75,
        ),
    ],
    ids=["standard", "openpbr"],
)
def test_pathtracer_material_guides_preserve_authored_base_color_and_metallic(
    device_type: spy.DeviceType,
    device: spy.Device,
    material_type: object,
    properties: dict[str, object],
    expected_color: np.ndarray,
    expected_metallic: float,
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(material_type, f2.Properties(properties))
    _add_quad(scene, z=0.0, normal_z=-1.0, material=material)
    scene.update()

    node = ReferencePathTracerNode.create(device)
    node.guide_output_specs = {
        "material_color": f2.ContainerSpec.auto(),
        "metallic": f2.ContainerSpec.auto(),
    }
    _, guides = node(scene, _make_quad_camera(scene), iteration=0)

    np.testing.assert_allclose(
        guides["material_color"].to_numpy()[4, 4, :3],
        expected_color,
        rtol=1e-3,
        atol=1e-3,
    )
    assert guides["metallic"].to_numpy()[4, 4] == pytest.approx(
        expected_metallic,
        abs=1e-3,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_guide_specs_respect_user_texture_spec(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """Guide outputs resolve user texture specs without DLSS-specific restrictions."""
    node = ReferencePathTracerNode.create(device)
    node.guide_output_specs = {
        "depth": f2.ContainerSpec.texture2d(spy.Format.r32_float),
    }
    cam = helpers.create_test_camera(helmet_scene, width=8, height=8, fov_y=45)

    _, guides = node(helmet_scene, cam, iteration=0)

    assert isinstance(guides["depth"], spy.Texture)
    assert guides["depth"].format == spy.Format.r32_float
    assert guides["depth"].width == 8
    assert guides["depth"].height == 8


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pathtracer_dlss_rr_guides_with_geometry(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """ReferencePathTracerNode renders the DLSS-RR guide subset with valid dimensions."""
    scene = _load_guide_test_scene(device)
    node = ReferencePathTracerNode.create(device)
    node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba16_float)
    guide_output_specs = {
        "diffuse_albedo": f2.ContainerSpec.auto(),
        "material_color": f2.ContainerSpec.auto(),
        "specular_albedo": f2.ContainerSpec.auto(),
        "normals": f2.ContainerSpec.auto(),
        "roughness": f2.ContainerSpec.auto(),
        "metallic": f2.ContainerSpec.auto(),
        "depth": f2.ContainerSpec.auto(),
        "specular_hit_distance": f2.ContainerSpec.auto(),
        "motion_vectors": f2.ContainerSpec.auto(),
    }
    node.guide_output_specs = guide_output_specs
    camera_position = spy.float3(0.0, 0.0, 4.0)
    cam = helpers.create_test_camera(
        scene,
        width=32,
        height=24,
        fov_y=45,
        position=camera_position,
        rotation=spy.math.quat_from_look_at(
            -spy.math.normalize(camera_position),
            spy.float3(0.0, 1.0, 0.0),
        ),
    )
    color_output, guides = node(
        scene,
        cam,
        iteration=0,
        subpixel_offset=spy.float2(0.0, 0.0),
        subpixel_random_jitter=0.0,
    )
    _, jittered_guides = node(
        scene,
        cam,
        iteration=1,
        subpixel_offset=spy.float2(0.25, -0.125),
        subpixel_random_jitter=0.0,
    )

    assert node._render_func is not None
    assert set(guides) == {
        "diffuse_albedo",
        "material_color",
        "specular_albedo",
        "normals",
        "roughness",
        "metallic",
        "depth",
        "hardware_depth",
        "specular_hit_distance",
        "motion_vectors",
        "emission",
        "geometry_id",
        "nan_count",
    }
    assert set(jittered_guides) == set(guides)
    for name in guide_output_specs:
        texture = guides[name]
        assert isinstance(texture, spy.Texture)
        assert texture.width == 32
        assert texture.height == 24
        data = texture.to_numpy()
        assert np.isfinite(data).all()
    assert guides["emission"] is None
    assert guides["geometry_id"] is None
    assert guides["hardware_depth"] is None
    assert guides["nan_count"] is None

    color = color_output.to_numpy()
    normals = guides["normals"].to_numpy()
    roughness = guides["roughness"].to_numpy()
    depth = guides["depth"].to_numpy()
    specular_hit_distance = guides["specular_hit_distance"].to_numpy()
    motion_vectors = guides["motion_vectors"].to_numpy()

    assert color.shape == (24, 32, 4)
    assert color[..., :3].max() > 0.0
    assert np.abs(normals[..., :3]).max() > 0.0
    assert roughness.min() >= 0.0
    assert roughness.max() <= 1.0
    assert depth.max() >= 0.0
    assert specular_hit_distance.min() >= 0.0
    assert specular_hit_distance.max() > 0.0
    assert np.abs(motion_vectors).max() < 1e-3

    previous_camera_position = spy.float3(0.2, 0.0, 4.0)
    previous_cam = helpers.create_test_camera(
        scene,
        width=32,
        height=24,
        fov_y=45,
        position=previous_camera_position,
        rotation=spy.math.quat_from_look_at(
            -spy.math.normalize(previous_camera_position),
            spy.float3(0.0, 1.0, 0.0),
        ),
    )
    node.reset()
    node(
        scene,
        previous_cam,
        iteration=2,
        subpixel_offset=spy.float2(0.0, 0.0),
        subpixel_random_jitter=0.0,
    )
    _, moved_guides = node(
        scene,
        cam,
        iteration=3,
        subpixel_offset=spy.float2(0.0, 0.0),
        subpixel_random_jitter=0.0,
    )
    moved_depth = moved_guides["depth"].to_numpy()
    moved_motion_vectors = moved_guides["motion_vectors"].to_numpy()
    hit_mask = moved_depth > 0.0

    assert hit_mask.any()
    assert np.abs(moved_motion_vectors[hit_mask]).max() > 0.01


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_runtime_background_settings(device_type: spy.DeviceType, device: spy.Device) -> None:
    """Primary misses use black or the configured runtime background color."""
    scene = f2.Scene.load(device, "data/assets/kronos/Avocado/glTF-Binary/Avocado.glb")
    light_entity = scene.create_entity()
    light_entity.create_component(f2.PointLight)
    scene.update()

    camera = helpers.create_test_camera(
        scene,
        width=8,
        height=8,
        fov_y=45,
        position=spy.float3(0.0, 0.0, 6.0),
        rotation=spy.math.quat_from_look_at(spy.float3(0.0, 0.0, 1.0), spy.float3(0.0, 1.0, 0.0)),
    )
    node = ReferencePathTracerNode.create(device)
    node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    node.use_background_color = False
    black_image, _ = node(scene, camera, iteration=0)
    black_image = black_image.to_numpy()
    render_func = node._render_func

    np.testing.assert_allclose(black_image[..., :3], 0.0, atol=1e-6)

    node.max_depth = 4
    node.use_background_color = True
    node.background_color = spy.float3(0.1, 0.2, 0.3)

    image, _ = node(scene, camera, iteration=1)
    image = image.to_numpy()

    assert node._render_func is render_func
    np.testing.assert_allclose(image[..., 0], 0.1, atol=1e-4)
    np.testing.assert_allclose(image[..., 1], 0.2, atol=1e-4)
    np.testing.assert_allclose(image[..., 2], 0.3, atol=1e-4)
    np.testing.assert_allclose(image[..., 3], 1.0, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_shader_helper(
    device_type: spy.DeviceType, device: spy.Device, empty_scene: f2.Scene
) -> None:
    """SceneShaderHelper loads and caches modules linked with scene."""
    from falcor2.editor import SceneShaderHelper

    helper = SceneShaderHelper(device)
    mod1 = helper.get_module(empty_scene, "falcor2.rendernodes.reference_pathtracer")
    mod2 = helper.get_module(empty_scene, "falcor2.rendernodes.reference_pathtracer")
    assert mod1 is mod2


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_shader_helper_accepts_loaded_module(
    device_type: spy.DeviceType, device: spy.Device, empty_scene: f2.Scene
) -> None:
    """SceneShaderHelper accepts a preloaded module and caches the linked result."""
    from falcor2.editor import SceneShaderHelper

    module = spy.Module(device.load_module("falcor2.rendernodes.reference_pathtracer"))
    helper = SceneShaderHelper(device)

    mod1 = helper.get_module(empty_scene, module)
    mod2 = helper.get_module(empty_scene, module)

    assert mod1 is mod2
    assert mod1.layout.find_type_by_name(WRITE_GUIDE_INTERFACE) is not None
