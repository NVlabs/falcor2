# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
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
    point_light.intensity = spy.float3(1.0)

    environment_entity = scene.create_entity()
    environment_light = environment_entity.create_component(f2.ConstantLight)
    environment_light.radiance = spy.float3(2.0)
    return point_light, environment_light


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_uniform_sampler_environment_pdf(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    scene.update()

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module = helper.get_module(scene, source_module)
    direction_probe = (
        module.environment_direction_pdf.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    directions = spy.Tensor.from_numpy(device, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

    # No environment lights evaluate to zero without accessing a distribution.
    np.testing.assert_allclose(direction_probe(directions).to_numpy(), 0.0, atol=0.0)

    entity = scene.create_entity()
    light = entity.create_component(f2.ConstantLight)
    light.radiance = spy.float3(1.0)
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
    distant_light.radiance = spy.float3(1.0)
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
    light.radiance = spy.float3(0.0)
    distant_light.radiance = spy.float3(0.0)
    scene.update()
    np.testing.assert_allclose(
        direction_probe(directions).to_numpy(),
        np.array([0.625 / np.pi, 0.125 / np.pi], dtype=np.float32),
        rtol=1e-6,
    )


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

    point_light.intensity = spy.float3(4.0)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 3

    environment_light.radiance = spy.float3(4.0)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 4

    transform = f2.Transform()
    transform.translation = spy.float3(2.0, 0.0, 0.0)
    entity.transform = transform
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 5

    material.emissive_factor = spy.float3(2.0)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 6


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
    module = helper.get_module(scene, source_module)

    def bind_uniform(cursor: Any) -> None:
        helper.bind_scene(cursor)

    def bind_power(cursor: Any) -> None:
        helper.bind_scene(cursor)
        cursor["power_light_sampler"] = power_sampler

    uniform_source_pdf = (
        module.uniform_source_pdf.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(bind_uniform)
        .return_type(spy.Tensor)
    )
    power_source_pdf = (
        module.power_source_pdf.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(bind_power)
        .return_type(spy.Tensor)
    )
    uniform_emissive_pdf = (
        module.uniform_emissive_pdf.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(bind_uniform)
        .return_type(spy.Tensor)
    )
    power_emissive_pdf = (
        module.power_emissive_pdf.as_func()
        .type_conformances(scene.requirements.type_conformances)
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
    environment_light.radiance = spy.float3(200.0)
    scene.update()
    assert power_sampler.update(scene)
    np.testing.assert_allclose(
        power_source_pdf(emitter_types, source_indices).to_numpy(),
        [0.25, 0.5, 0.125, 0.125],
        rtol=1e-5,
    )

    # When only the finite domain has a positive estimate, it receives the
    # complete selection probability and remains power-weighted internally.
    environment_light.radiance = spy.float3(0.0)
    scene.update()
    assert power_sampler.update(scene)
    np.testing.assert_allclose(
        power_source_pdf(emitter_types, source_indices).to_numpy(),
        [0.5, 0.0, 0.25, 0.25],
        rtol=1e-5,
        atol=0.0,
    )

    # Likewise, a sole positive infinite domain receives probability one.
    environment_light.radiance = spy.float3(2.0)
    point_light.intensity = spy.float3(0.0)
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
    environment_light.radiance = spy.float3(0.0)
    scene.update()
    assert power_sampler.update(scene)
    np.testing.assert_allclose(
        power_source_pdf(one_domain_types, one_domain_indices).to_numpy(),
        0.0,
        rtol=1e-6,
        atol=0.0,
    )


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
