# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from pathlib import Path
import weakref
from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
from falcor2.editor import SceneShaderHelper
import falcor2.testing.helpers as helpers
from falcor2.rendernodes import ReferencePathTracerNode


def _add_quad(
    scene: f2.Scene,
    material: f2.Material,
    *,
    z: float = 0.0,
    normal_z: float = 1.0,
    size: float = 2.0,
) -> f2.Entity:
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    half_size = 0.5 * size
    geometry.set_mesh_data(
        positions=np.array(
            [
                [-half_size, -half_size, z],
                [half_size, -half_size, z],
                [-half_size, half_size, z],
                [half_size, half_size, z],
            ],
            dtype=np.float32,
        ),
        sub_mesh_indices=[
            (
                np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)
                if normal_z > 0.0
                else np.array([[0, 2, 1], [2, 3, 1]], dtype=np.uint32)
            ),
        ],
        normals=np.tile(np.array([0.0, 0.0, normal_z], dtype=np.float32), (4, 1)),
        name="quad",
    )
    entity = scene.create_entity()
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]
    return entity


def _add_emissive_quad(scene: f2.Scene) -> tuple[f2.StandardMaterial, f2.Entity]:
    material = scene.create_material(
        f2.StandardMaterial,
        f2.Properties({"emissive_factor": spy.float3(1.0)}),
    )
    return material, _add_quad(scene, material)


def _add_component_lights(scene: f2.Scene) -> tuple[f2.PointLight, f2.ConstantLight]:
    point_entity = scene.create_entity()
    point_light = point_entity.create_component(f2.PointLight)
    point_light.intensity = 1.0

    environment_entity = scene.create_entity()
    environment_light = environment_entity.create_component(f2.ConstantLight)
    environment_light.intensity = 2.0
    return point_light, environment_light


def _make_light_power_probe(scene: f2.Scene, device: spy.Device) -> Any:
    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module, requirements = helper.prepare(scene, source_module)
    return (
        module.component_light_power.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "light_type, scale, expected_world_space_area",
    [
        (f2.SphereLight, (2.0, 3.0, 4.0), np.pi * 24.0 ** (2.0 / 3.0)),
        (f2.CylinderLight, (2.0, 3.0, 4.0), 2.0 * np.pi * np.sqrt(12.0)),
        (f2.DiskLight, (2.0, 3.0, 4.0), 1.5 * np.pi),
        (f2.RectLight, (2.0, 3.0, 4.0), 6.0),
    ],
)
def test_normalized_area_light_power_is_independent_of_world_space_area(
    device_type: spy.DeviceType,
    device: spy.Device,
    light_type: type[f2.Light],
    scale: tuple[float, float, float],
    expected_world_space_area: float,
) -> None:
    scene = f2.Scene.create(device)
    entity = scene.create_entity()
    light = entity.create_component(light_type)
    light.intensity = 2.0
    light.exposure = 1.0
    light.normalize = True
    scene.update()

    light_power = _make_light_power_probe(scene, device)
    light_ids = spy.Tensor.from_numpy(device, np.array([0], dtype=np.uint32))

    unit_scale_power = light_power(light_ids).to_numpy()
    np.testing.assert_allclose(unit_scale_power, 4.0 * np.pi, rtol=1e-6)

    transform = f2.Transform()
    transform.scale = spy.float3(*scale)
    entity.transform = transform
    scene.update()
    double_scale_power = light_power(light_ids).to_numpy()
    np.testing.assert_allclose(double_scale_power, unit_scale_power, rtol=1e-6)

    light.normalize = False
    scene.update()
    unnormalized_power = light_power(light_ids).to_numpy()
    np.testing.assert_allclose(
        unnormalized_power / double_scale_power, expected_world_space_area, rtol=1e-6
    )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_normalized_distant_light_uses_usd_projected_angular_factor(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    entity = scene.create_entity()
    light = entity.create_component(f2.DistantLight)
    light.intensity = 2.0
    light.exposure = 1.0
    light.normalize = True
    light.cutoff_angle = 30.0
    scene.update()

    light_power = _make_light_power_probe(scene, device)
    light_ids = spy.Tensor.from_numpy(device, np.array([0], dtype=np.uint32))
    half_angle = np.deg2rad(light.cutoff_angle)
    solid_angle = 2.0 * np.pi * (1.0 - np.cos(half_angle))
    projected_factor = np.pi * np.sin(half_angle) ** 2
    np.testing.assert_allclose(
        light_power(light_ids).to_numpy(),
        4.0 * solid_angle / projected_factor,
        rtol=1e-6,
    )

    light.normalize = False
    scene.update()
    np.testing.assert_allclose(light_power(light_ids).to_numpy(), 4.0 * solid_angle, rtol=1e-6)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_normalize_is_a_noop_for_constant_lights(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    entity = scene.create_entity()
    light = entity.create_component(f2.ConstantLight)
    light.intensity = 2.0
    light.exposure = 1.0
    light.normalize = True
    scene.update()

    light_power = _make_light_power_probe(scene, device)
    light_ids = spy.Tensor.from_numpy(device, np.array([0], dtype=np.uint32))
    normalized_power = light_power(light_ids).to_numpy()
    np.testing.assert_allclose(normalized_power, 16.0 * np.pi, rtol=1e-6)

    light.normalize = False
    scene.update()
    np.testing.assert_allclose(light_power(light_ids).to_numpy(), normalized_power, rtol=1e-6)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_environment_map_power_rebuilds_after_texture_replacement(
    device_type: spy.DeviceType,
    device: spy.Device,
    tmp_path: Path,
) -> None:
    radiance = 2.0
    paths = [tmp_path / "environment.exr", tmp_path / "brighter_environment.exr"]
    for path, scale in zip(paths, (1.0, 2.0), strict=True):
        spy.Bitmap(
            np.full((4, 8, 3), radiance * scale, dtype=np.float32),
            srgb_gamma=False,
        ).write(path)

    scene = f2.Scene.create(device)
    light = scene.create_entity().create_component(f2.EnvMapLight)
    light.intensity = 1.0
    light.env_map_path = paths[0]
    scene.update()

    sampler = f2.PowerLightSampler()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 1
    light_power = _make_light_power_probe(scene, device)
    light_ids = spy.Tensor.from_numpy(device, np.array([0], dtype=np.uint32))
    original_power = light_power(light_ids).to_numpy()
    np.testing.assert_allclose(original_power, 4.0 * np.pi * radiance, rtol=1e-6)

    light.env_map_path = paths[1]
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 2
    np.testing.assert_allclose(light_power(light_ids).to_numpy(), 2.0 * original_power, rtol=1e-6)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_uniform_sampler_environment_pdf(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    scene.update()

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module, requirements = helper.prepare(scene, source_module)
    direction_probe = (
        module.environment_direction_pdf.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    directions = spy.Tensor.from_numpy(device, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

    # No environment lights evaluate to zero without accessing a distribution.
    np.testing.assert_allclose(direction_probe(directions).to_numpy(), 0.0, atol=0.0)

    entity = scene.create_entity()
    light = entity.create_component(f2.ConstantLight)
    light.intensity = 1.0
    scene.update()

    # A single environment light delegates directly to the light's PDF without
    # constructing or binding a selection distribution.
    np.testing.assert_allclose(
        direction_probe(directions).to_numpy(),
        np.array([1.0 / (4.0 * np.pi)], dtype=np.float32),
        rtol=1e-6,
    )

    distant_entity = scene.create_entity()
    distant_light = distant_entity.create_component(f2.DistantLight)
    distant_light.intensity = 1.0
    distant_light.cutoff_angle = 60.0

    scene.update()

    directions = spy.Tensor.from_numpy(
        device,
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
    )
    # Each environment source has probability 1/2. The first direction is
    # inside the distant-light cone; the second only has the constant-light PDF.
    np.testing.assert_allclose(
        direction_probe(directions).to_numpy(),
        np.array([0.625 / np.pi, 0.125 / np.pi], dtype=np.float32),
        rtol=1e-6,
    )

    # Uniform selection is independent of emitted radiance.
    light.intensity = 0.0
    distant_light.intensity = 0.0
    scene.update()
    np.testing.assert_allclose(
        direction_probe(directions).to_numpy(),
        np.array([0.625 / np.pi, 0.125 / np.pi], dtype=np.float32),
        rtol=1e-6,
    )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_power_sampler_rebuilds_for_all_light_sources(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    material, entity = _add_emissive_quad(scene)
    sampler = f2.PowerLightSampler()

    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 1
    assert not sampler.update(scene)

    point_light, environment_light = _add_component_lights(scene)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 2

    point_light.intensity = 4.0
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 3

    environment_light.intensity = 4.0
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 4

    transform = f2.Transform()
    transform.translation = spy.float3(2.0, 0.0, 0.0)
    entity.transform = transform
    scene.update()
    assert not sampler.update(scene)
    assert sampler.rebuild_count == 4

    material.emissive_factor = spy.float3(2.0)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 5


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_samplers_track_only_potentially_emissive_entities(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    _add_emissive_quad(scene)
    material = scene.create_material(f2.StandardMaterial)
    entity = _add_quad(scene, material)
    scene.update()
    power_sampler = f2.PowerLightSampler()
    hierarchical_sampler = f2.HierarchicalLightSampler()
    assert power_sampler.update(scene)
    assert hierarchical_sampler.update(scene)

    transform = f2.Transform()
    transform.scale = spy.float3(2.0, 3.0, 4.0)
    entity.transform = transform
    scene.update()
    assert not power_sampler.update(scene)
    assert not hierarchical_sampler.update(scene)

    # Material edits must refresh membership even when instance assignments stay unchanged.
    material.emissive_factor = spy.float3(1.0)
    scene.update()
    assert power_sampler.update(scene)
    assert hierarchical_sampler.update(scene)
    transform.scale = spy.float3(4.0, 6.0, 8.0)
    entity.transform = transform
    scene.update()
    assert power_sampler.update(scene)
    assert hierarchical_sampler.update(scene)
    material.emissive_factor = spy.float3(0.0)
    scene.update()
    assert power_sampler.update(scene)
    assert hierarchical_sampler.update(scene)
    transform.scale = spy.float3(2.0)
    entity.transform = transform
    scene.update()
    assert not power_sampler.update(scene)
    assert not hierarchical_sampler.update(scene)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_samplers_distinguish_rigid_motion_from_scale_changes(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    _, entity = _add_emissive_quad(scene)
    transform = f2.Transform()
    transform.scale = spy.float3(2.0, 3.0, 4.0)
    entity.transform = transform
    scene.update()

    power_sampler = f2.PowerLightSampler()
    hierarchical_sampler = f2.HierarchicalLightSampler()
    assert power_sampler.update(scene)
    assert hierarchical_sampler.update(scene)

    transform.translation = spy.float3(2.0, -1.0, 3.0)
    transform.rotation_xyz = spy.float3(0.3, 0.7, -0.4)
    entity.transform = transform
    scene.update()
    assert not power_sampler.update(scene)
    assert power_sampler.rebuild_count == 1
    assert hierarchical_sampler.update(scene)
    assert hierarchical_sampler.rebuild_count == 1
    assert hierarchical_sampler.refit_count == 1

    transform.scale *= 2.0
    entity.transform = transform
    scene.update()
    assert power_sampler.update(scene)
    assert power_sampler.rebuild_count == 2
    assert hierarchical_sampler.update(scene)
    assert hierarchical_sampler.rebuild_count == 1
    assert hierarchical_sampler.refit_count == 2


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_power_sampler_preserves_weights_through_animated_rotation_hierarchy(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    _, emitter = _add_emissive_quad(scene)
    transform = f2.Transform()
    transform.scale = spy.float3(2.0, 3.0, 4.0)
    emitter.transform = transform
    child = emitter
    # Composing interpolated float rotations introduces more roundoff than a single rotation.
    for level in range(4):
        parent = scene.create_entity()
        child.parent = parent
        child = parent
        channel = f2.AnimationChannel()
        channel.value_type = f2.AnimationValueType.quatf
        channel.interpolation_mode = f2.AnimationInterpolationMode.linear
        channel.times = [0.0, 1.0, 2.0]
        values = []
        for key in range(3):
            rotation = f2.Transform()
            rotation.rotation_xyz = spy.float3(0.7 * (level + key), -0.9 * (level - key), 1.3 * key)
            values.extend(list(rotation.rotation))
        channel.values = values
        animation = scene.create_animation()
        animation.channels = [channel]
        component = parent.create_component(f2.TransformAnimation)
        component.animation = animation
        component.rotation_channel = 0

    scene.update()
    sampler = f2.PowerLightSampler()
    assert sampler.update(scene)
    for frame in range(1, 121):
        scene.time = frame / 60.0
        scene.update()
        assert not sampler.update(scene), f"Unexpected rebuild at frame {frame}"
    assert sampler.rebuild_count == 1

    # Real scale changes still invalidate the distribution after the rotation sweep.
    transform.scale *= 1.001
    emitter.transform = transform
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 2


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_uniform_and_power_sampler_cover_all_light_sources(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    point_light, environment_light = _add_component_lights(scene)
    emissive_material, _ = _add_emissive_quad(scene)
    scene.update()

    power_sampler = f2.PowerLightSampler()
    assert power_sampler.update(scene)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module, requirements = helper.prepare(scene, source_module)

    def bind_uniform(cursor: Any) -> None:
        helper.bind_scene(cursor)

    def bind_power(cursor: Any) -> None:
        helper.bind_scene(cursor)
        cursor["power_light_sampler"] = power_sampler

    uniform_source_pdf = (
        module.uniform_source_pdf.as_func()
        .type_conformances(requirements.type_conformances)
        .write(bind_uniform)
        .return_type(spy.Tensor)
    )
    power_source_pdf = (
        module.power_source_pdf.as_func()
        .type_conformances(requirements.type_conformances)
        .write(bind_power)
        .return_type(spy.Tensor)
    )
    uniform_emissive_pdf = (
        module.uniform_emissive_pdf.as_func()
        .type_conformances(requirements.type_conformances)
        .write(bind_uniform)
        .return_type(spy.Tensor)
    )
    power_emissive_pdf = (
        module.power_emissive_pdf.as_func()
        .type_conformances(requirements.type_conformances)
        .write(bind_power)
        .return_type(spy.Tensor)
    )

    emitter_types = spy.Tensor.from_numpy(
        device,
        np.array([1, 2, 3, 3], dtype=np.uint32),
    )
    source_indices = spy.Tensor.from_numpy(
        device,
        np.array([0, 0, 0, 1], dtype=np.uint32),
    )

    uniform_pdf = uniform_source_pdf(emitter_types, source_indices).to_numpy()
    np.testing.assert_allclose(uniform_pdf, 0.25, rtol=1e-6)

    power_pdf = power_source_pdf(emitter_types, source_indices).to_numpy()
    np.testing.assert_allclose(power_pdf, [0.25, 0.5, 0.125, 0.125], rtol=1e-5)

    # Stale and invalid triangle IDs must be rejected before accessing the
    # triangle-to-active-triangle mapping.
    invalid_triangle_ids = spy.Tensor.from_numpy(
        device,
        np.array([2, np.iinfo(np.uint32).max], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        uniform_emissive_pdf(invalid_triangle_ids).to_numpy(), np.zeros(2, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        power_emissive_pdf(invalid_triangle_ids).to_numpy(), np.zeros(2, dtype=np.float32)
    )

    # Finite and infinite emitters retain a fixed 50/50 domain split even when
    # their estimates have vastly different magnitudes.
    environment_light.intensity = 200.0
    scene.update()
    assert power_sampler.update(scene)
    np.testing.assert_allclose(
        power_source_pdf(emitter_types, source_indices).to_numpy(),
        [0.25, 0.5, 0.125, 0.125],
        rtol=1e-5,
    )

    # When only the finite domain has a positive estimate, it receives the
    # complete selection probability and remains power-weighted internally.
    environment_light.intensity = 0.0
    scene.update()
    assert power_sampler.update(scene)
    np.testing.assert_allclose(
        power_source_pdf(emitter_types, source_indices).to_numpy(),
        [0.5, 0.0, 0.25, 0.25],
        rtol=1e-5,
        atol=0.0,
    )

    # Likewise, a sole positive infinite domain receives probability one.
    environment_light.intensity = 2.0
    point_light.intensity = 0.0
    emissive_material.emissive_factor = spy.float3(0.0)
    scene.update()
    assert power_sampler.update(scene)
    one_domain_types = spy.Tensor.from_numpy(
        device,
        np.array([1, 2], dtype=np.uint32),
    )
    one_domain_indices = spy.Tensor.from_numpy(
        device,
        np.array([0, 0], dtype=np.uint32),
    )
    np.testing.assert_allclose(
        power_source_pdf(one_domain_types, one_domain_indices).to_numpy(),
        [0.0, 1.0],
        rtol=1e-6,
        atol=0.0,
    )

    # If every present source has zero power, PowerLightSampler reports no
    # sampleable sources.
    environment_light.intensity = 0.0
    scene.update()
    assert power_sampler.update(scene)
    np.testing.assert_allclose(
        power_source_pdf(one_domain_types, one_domain_indices).to_numpy(),
        0.0,
        rtol=1e-6,
        atol=0.0,
    )


@pytest.mark.nightly_if(device_type=(spy.DeviceType.cuda,))
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_area_light_geometry_and_mis_pdfs(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    scene.update()
    empty_requirements_generation = scene.requirements_generation
    point_entity = scene.create_entity()
    point_transform = f2.Transform()
    point_transform.translation = spy.float3(100.0, 0.0, 0.0)
    point_entity.transform = point_transform
    point_light = point_entity.create_component(f2.PointLight)
    light_types = (f2.SphereLight, f2.DiskLight, f2.RectLight, f2.CylinderLight)
    x_positions = (-3.0, 0.0, 3.0, 6.0)
    light_entities = []
    lights = []
    for light_type, x in zip(light_types, x_positions, strict=True):
        entity = scene.create_entity()
        transform = f2.Transform()
        transform.translation = spy.float3(x, 0.0, 0.0)
        entity.transform = transform
        light_entities.append(entity)
        light = entity.create_component(light_type)
        if isinstance(light, (f2.SphereLight, f2.CylinderLight, f2.DiskLight)):
            light.radius = 1.0
        lights.append(light)
    lights[0].enable_virtual_sphere_shrinking = True

    scene.update()
    assert scene.requirements_generation > empty_requirements_generation
    assert scene.requirements.has_geometry_type(f2.GeometryType.area_light)
    power_sampler = f2.PowerLightSampler()
    hierarchical_sampler = f2.HierarchicalLightSampler()
    assert power_sampler.update(scene)
    assert hierarchical_sampler.update(scene)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module, requirements = helper.prepare(scene, source_module)

    def bind_uniform(cursor: Any) -> None:
        helper.bind_scene(cursor)

    def bind_power(cursor: Any) -> None:
        helper.bind_scene(cursor)
        cursor["power_light_sampler"] = power_sampler

    def bind_hierarchical(cursor: Any) -> None:
        helper.bind_scene(cursor)
        cursor["hierarchical_light_sampler"] = hierarchical_sampler

    uniform_pdf = (
        module.uniform_area_light_pdf.as_func()
        .type_conformances(requirements.type_conformances)
        .write(bind_uniform)
        .return_type(spy.Tensor)
    )
    power_pdf = (
        module.power_area_light_pdf.as_func()
        .type_conformances(requirements.type_conformances)
        .write(bind_power)
        .return_type(spy.Tensor)
    )
    hierarchical_pdf = (
        module.hierarchical_area_light_pdf.as_func()
        .type_conformances(requirements.type_conformances)
        .write(bind_hierarchical)
        .return_type(spy.Tensor)
    )
    trace = (
        module.trace_area_light.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    trace_visible = (
        module.trace_area_light_visible.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )

    # The leading point light has LightID 0 but no procedural instance. This
    # exercises the per-instance LightID payload for IDs 1 through 4.
    light_ids = spy.Tensor.from_numpy(device, np.arange(1, 5, dtype=np.uint32))
    receiver_positions_np = np.array(
        [[-3.0, 0.0, 3.0], [0.0, 0.0, 2.0], [3.0, 0.0, 2.0], [6.0, 0.0, -3.0]],
        dtype=np.float32,
    )
    emitter_positions_np = np.array(
        [[-3.0, 0.0, 1.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    normals_np = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    receiver_positions = spy.Tensor.from_numpy(device, receiver_positions_np)
    emitter_positions = spy.Tensor.from_numpy(device, emitter_positions_np)
    normals = spy.Tensor.from_numpy(device, normals_np)

    # Uniform selection contributes 1/5. The sphere uses uniform solid-angle
    # cone sampling; the other area lights use uniform surface sampling.
    sphere_cos_cone_angle = np.sqrt(1.0 - 1.0 / 9.0)
    sphere_pdf = 1.0 / (2.0 * np.pi * (1.0 - sphere_cos_cone_angle))
    np.testing.assert_allclose(
        uniform_pdf(light_ids, receiver_positions, emitter_positions, normals).to_numpy(),
        np.array(
            [sphere_pdf / 5.0, 4.0 / (5.0 * np.pi), 4.0 / 5.0, 2.0 / (5.0 * np.pi)],
            dtype=np.float32,
        ),
        rtol=1e-5,
    )
    near_sphere_receiver = spy.Tensor.from_numpy(
        device, np.array([[-3.0, 0.0, 1.001]], dtype=np.float32)
    )
    sphere_id = spy.Tensor.from_numpy(device, np.array([1], dtype=np.uint32))
    sphere_emitter = spy.Tensor.from_numpy(device, np.array([[-3.0, 0.0, 1.0]], dtype=np.float32))
    sphere_normal = spy.Tensor.from_numpy(device, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(
        uniform_pdf(sphere_id, near_sphere_receiver, sphere_emitter, sphere_normal).to_numpy(),
        0.0,
        atol=0.0,
    )
    power_values = power_pdf(light_ids, receiver_positions, emitter_positions, normals).to_numpy()
    hierarchical_values = hierarchical_pdf(
        light_ids, receiver_positions, emitter_positions, normals
    ).to_numpy()
    assert np.all(power_values > 0.0)
    np.testing.assert_allclose(hierarchical_values, power_values, rtol=1e-6)

    # CUDA compute kernels cannot issue inline ray queries; its OptiX tracing
    # path is covered by the reference-path-tracer test below.
    if device_type != spy.DeviceType.cuda:
        ray_origins = spy.Tensor.from_numpy(
            device,
            np.array(
                [[-3.0, 0.0, 3.0], [0.0, 0.0, -2.0], [3.0, 0.0, -2.0], [6.0, 0.0, -2.0]],
                dtype=np.float32,
            ),
        )
        ray_directions = spy.Tensor.from_numpy(
            device,
            np.array(
                [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        )
        all_instance_inclusion_masks = spy.Tensor.from_numpy(
            device, np.full(4, 0x07, dtype=np.uint32)
        )
        primary_instance_inclusion_masks = spy.Tensor.from_numpy(
            device, np.full(4, 0x05, dtype=np.uint32)
        )
        visibility_instance_inclusion_masks = spy.Tensor.from_numpy(
            device, np.full(4, 0x05, dtype=np.uint32)
        )
        secondary_instance_inclusion_masks = spy.Tensor.from_numpy(
            device, np.full(4, 0x03, dtype=np.uint32)
        )
        regular_geometry_instance_inclusion_masks = spy.Tensor.from_numpy(
            device, np.full(4, 0x01, dtype=np.uint32)
        )
        all_hits = trace(ray_origins, ray_directions, all_instance_inclusion_masks).to_numpy()
        np.testing.assert_allclose(all_hits[:, 0], 1.0, atol=0.0)
        np.testing.assert_allclose(all_hits[:, 1], 2.0, atol=0.0)
        np.testing.assert_allclose(all_hits[:, 2], [2.0, 2.0, 2.0, 1.0], rtol=1e-5)
        np.testing.assert_allclose(all_hits[:, 3], [1.0, 2.0, 3.0, 4.0], atol=0.0)
        secondary_hits = trace(
            ray_origins, ray_directions, secondary_instance_inclusion_masks
        ).to_numpy()
        np.testing.assert_allclose(secondary_hits, all_hits, rtol=1e-5, atol=0.0)
        regular_geometry_hits = trace(
            ray_origins, ray_directions, regular_geometry_instance_inclusion_masks
        ).to_numpy()
        np.testing.assert_allclose(regular_geometry_hits[:, 0], 0.0, atol=0.0)

        for geometry_visible in (False, True):
            lights[1].geometry_visible = geometry_visible
            scene.update()

            primary_hits = trace(
                ray_origins, ray_directions, primary_instance_inclusion_masks
            ).to_numpy()
            visibility_hits = trace(
                ray_origins, ray_directions, visibility_instance_inclusion_masks
            ).to_numpy()
            secondary_hits = trace(
                ray_origins, ray_directions, secondary_instance_inclusion_masks
            ).to_numpy()
            payload_visibility = trace_visible(
                ray_origins, ray_directions, secondary_instance_inclusion_masks
            ).to_numpy()

            np.testing.assert_allclose(
                primary_hits[:, 0], [0.0, float(geometry_visible), 0.0, 0.0], atol=0.0
            )
            np.testing.assert_allclose(
                visibility_hits[:, 0], [0.0, float(geometry_visible), 0.0, 0.0], atol=0.0
            )
            np.testing.assert_allclose(secondary_hits, all_hits, rtol=1e-5, atol=0.0)
            np.testing.assert_array_equal(payload_visibility, [0, int(geometry_visible), 0, 0])

        lights[1].geometry_visible = False
        scene.update()

        # Hidden planar lights are one-sided traversal events. Their non-emitting
        # back faces are rejected, while a visible surface remains a two-sided blocker.
        back_origins = spy.Tensor.from_numpy(
            device,
            np.array(
                [[-3.0, 0.0, -2.0], [0.0, 0.0, 2.0], [3.0, 0.0, 2.0], [6.0, 0.0, 2.0]],
                dtype=np.float32,
            ),
        )
        back_directions = spy.Tensor.from_numpy(
            device,
            np.array(
                [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
                dtype=np.float32,
            ),
        )
        hidden_back_hits = trace(
            back_origins, back_directions, secondary_instance_inclusion_masks
        ).to_numpy()
        np.testing.assert_allclose(hidden_back_hits[:, 0], [1.0, 0.0, 0.0, 1.0], atol=0.0)
        lights[1].geometry_visible = True
        lights[2].geometry_visible = True
        scene.update()
        visible_back_hits = trace(
            back_origins, back_directions, secondary_instance_inclusion_masks
        ).to_numpy()
        np.testing.assert_allclose(visible_back_hits[:, 0], 1.0, atol=0.0)
        lights[1].geometry_visible = False
        lights[2].geometry_visible = False
        scene.update()

        moved_transform = f2.Transform()
        moved_transform.translation = spy.float3(4.0, 0.0, 0.0)
        light_entities[2].transform = moved_transform
        scene.update()
        moved_origin = spy.Tensor.from_numpy(device, np.array([[4.0, 0.0, -2.0]], dtype=np.float32))
        moved_direction = spy.Tensor.from_numpy(
            device, np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        )
        all_instance_inclusion_mask = spy.Tensor.from_numpy(
            device, np.array([0x07], dtype=np.uint32)
        )
        moved_hit = trace(moved_origin, moved_direction, all_instance_inclusion_mask).to_numpy()
        np.testing.assert_allclose(moved_hit[:, [0, 2, 3]], [[1.0, 2.0, 3.0]], rtol=1e-5)

        # Light IDs may change while the per-light render instances remain alive.
        point_light.active = False
        scene.update()
        remapped_hit = trace(moved_origin, moved_direction, all_instance_inclusion_mask).to_numpy()
        np.testing.assert_allclose(remapped_hit[:, [0, 2, 3]], [[1.0, 2.0, 2.0]], rtol=1e-5)

        populated_requirements_generation = scene.requirements_generation
        populated_requirements = scene.requirements
        for light in lights:
            light.active = False
        scene.update()
        assert scene.requirements_generation > populated_requirements_generation
        assert not scene.requirements.has_geometry_type(f2.GeometryType.area_light)
        assert scene.requirements != populated_requirements
        assert scene.requirements.matches_shader_inputs(populated_requirements)
        reused_module, updated_requirements = helper.prepare(scene, source_module)
        assert reused_module is module
        assert updated_requirements is scene.requirements
        removed_hit = trace(moved_origin, moved_direction, all_instance_inclusion_mask).to_numpy()
        np.testing.assert_allclose(removed_hit[:, 0], 0.0, atol=0.0)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_degenerate_area_lights_have_valid_geometry_proxies(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    sphere = scene.create_entity().create_component(f2.SphereLight)
    cylinder = scene.create_entity().create_component(f2.CylinderLight)
    disk = scene.create_entity().create_component(f2.DiskLight)
    rect = scene.create_entity().create_component(f2.RectLight)
    sphere.radius = 0.0
    cylinder.radius = 0.0
    cylinder.length = 0.0
    disk.radius = 0.0
    rect.width = 0.0
    rect.height = 0.0

    scene.update()

    assert scene.requirements.has_geometry_type(f2.GeometryType.area_light)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reference_pathtracer_exposes_light_sampler(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    node = ReferencePathTracerNode.create(device)
    assert isinstance(node.light_sampler, f2.PowerLightSampler)
    assert node.light_sampler.slang_type_name == "PowerLightSampler"

    replacement = f2.UniformLightSampler()
    node.light_sampler = replacement
    assert node.light_sampler is replacement

    with pytest.raises(TypeError):
        node.light_sampler = object()  # type: ignore[assignment]


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("sampler_type", [f2.UniformLightSampler, f2.PowerLightSampler])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reference_pathtracer_renders_with_light_sampler(
    sampler_type: type[f2.LightSampler],
    device_type: spy.DeviceType,
    device: spy.Device,
    helmet_scene: f2.Scene,
) -> None:
    node = ReferencePathTracerNode.create(device)
    node.light_sampler = sampler_type()
    camera = helpers.create_test_camera(helmet_scene, width=16, height=16, fov_y=45)
    color = spy.Tensor.empty(device, (16, 16), spy.float4)

    node._render(helmet_scene, camera, color, iteration=0)
    data = color.to_numpy()
    node_ref = weakref.ref(node)
    del node
    gc.collect()

    assert np.isfinite(data).all()
    assert data.max() > 0.0
    assert node_ref() is None
