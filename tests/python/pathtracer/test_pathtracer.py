# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers
import numpy as np
from falcor2.rendernodes import (
    ReferencePathTracerNode,
    WRITE_GUIDE_INTERFACE,
)


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
def test_pathtracer_settings_change(device_type: spy.DeviceType, device: spy.Device) -> None:
    """Changing settings updates constants dict."""
    pt = ReferencePathTracerNode.create(device)
    pt.enable_nee = True
    assert pt._constants["ENABLE_NEE"] == True
    pt.max_depth = 5
    assert pt._constants["MAX_DEPTH"] == 5


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
    assert first_previous["dims"] == spy.uint2(32, 24)
    assert isinstance(ray_samplers[-1], f2.CameraUniforms)
    assert ray_samplers[-1].width == 32
    assert ray_samplers[-1].height == 24

    node._render(empty_scene, camera, spy.Tensor.empty(device, (12, 16), spy.float4), 1)

    assert previous_cameras[-1]["dims"] == spy.uint2(16, 12)
    assert previous_cameras[-1] is not first_previous
    assert isinstance(previous_cameras[-1], dict)
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
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """ReferencePathTracerNode renders non-DLSS guide outputs."""
    node = ReferencePathTracerNode.create(device)
    node.guide_output_specs = {
        "material_color": f2.ContainerSpec.auto(),
        "emission": f2.ContainerSpec.auto(),
        "geometry_id": f2.ContainerSpec.auto(),
    }
    camera_position = spy.float3(0.0, 0.0, 4.0)
    cam = helpers.create_test_camera(
        helmet_scene,
        width=16,
        height=12,
        fov_y=45,
        position=camera_position,
        rotation=spy.math.quat_from_look_at(
            -spy.math.normalize(camera_position),
            spy.float3(0.0, 1.0, 0.0),
        ),
    )

    _, guides = node(helmet_scene, cam, iteration=0)

    assert isinstance(guides["material_color"], spy.Texture)
    assert isinstance(guides["emission"], spy.Texture)
    assert isinstance(guides["geometry_id"], spy.Texture)
    for name, value in guides.items():
        if name not in {"material_color", "emission", "geometry_id"}:
            assert value is None

    material_color = guides["material_color"].to_numpy()
    emission = guides["emission"].to_numpy()
    geometry_id = guides["geometry_id"].to_numpy()
    assert material_color.shape == (12, 16, 4)
    assert emission.shape == (12, 16, 4)
    assert geometry_id.shape == (12, 16, 4)
    assert np.isfinite(material_color).all()
    assert np.isfinite(emission).all()
    assert material_color[..., :3].max() > 0.0
    assert emission[..., :3].max() > 0.0
    valid_geometry = geometry_id[..., 1] != np.uint32(0xFFFFFFFF)
    assert valid_geometry.any()
    assert (geometry_id[..., 0][valid_geometry] == int(f2.GeometryType.triangle)).all()


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
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """ReferencePathTracerNode renders the DLSS-RR guide subset with valid dimensions."""
    node = ReferencePathTracerNode.create(device)
    node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba16_float)
    guide_output_specs = {
        "diffuse_albedo": f2.ContainerSpec.auto(),
        "material_color": f2.ContainerSpec.auto(),
        "specular_albedo": f2.ContainerSpec.auto(),
        "normals": f2.ContainerSpec.auto(),
        "roughness": f2.ContainerSpec.auto(),
        "depth": f2.ContainerSpec.auto(),
        "specular_hit_distance": f2.ContainerSpec.auto(),
        "motion_vectors": f2.ContainerSpec.auto(),
    }
    node.guide_output_specs = guide_output_specs
    camera_position = spy.float3(0.0, 0.0, 4.0)
    cam = helpers.create_test_camera(
        helmet_scene,
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
        helmet_scene,
        cam,
        iteration=0,
        subpixel_offset=spy.float2(0.0, 0.0),
        subpixel_random_jitter=0.0,
    )
    _, jittered_guides = node(
        helmet_scene,
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
        "depth",
        "hardware_depth",
        "specular_hit_distance",
        "motion_vectors",
        "emission",
        "geometry_id",
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
    assert np.abs(motion_vectors).max() < 1e-3

    previous_camera_position = spy.float3(0.2, 0.0, 4.0)
    previous_cam = helpers.create_test_camera(
        helmet_scene,
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
        helmet_scene,
        previous_cam,
        iteration=2,
        subpixel_offset=spy.float2(0.0, 0.0),
        subpixel_random_jitter=0.0,
    )
    _, moved_guides = node(
        helmet_scene,
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
def test_background_color_used_when_env_map_hidden(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Primary misses return the configured background color when the env map is hidden."""
    scene = f2.Scene(device, "data/assets/kronos/DamagedHelmet/glTF/DamagedHelmet.gltf")
    env_entity = scene.create_entity()
    env_map = env_entity.create_component(f2.EnvMapLight)
    env_map["env_map_path"] = "data/assets/envmaps/aerodynamics_workshop_512.hdr"
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
    node.env_map_as_background = False
    node.background_color = spy.float3(0.1, 0.2, 0.3)

    image, _ = node(scene, camera, iteration=0)
    image = image.to_numpy()

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
