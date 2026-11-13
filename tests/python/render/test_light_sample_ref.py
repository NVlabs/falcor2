# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
from falcor2.editor import SceneShaderHelper
import falcor2.testing.helpers as helpers


def _make_probes(
    device: spy.Device, tmp_path: Path, *, black_environment: bool = False
) -> tuple[Any, Any, f2.Scene]:
    scene = f2.Scene.create(device)
    for light_type in (
        f2.PointLight,
        f2.SphereLight,
        f2.DiskLight,
        f2.RectLight,
        f2.CylinderLight,
        f2.ConstantLight,
        f2.DistantLight,
    ):
        light = scene.create_entity().create_component(light_type)
        light.intensity = 2.0
        if isinstance(light, f2.DistantLight):
            light.cutoff_angle = 30.0
        if isinstance(light, f2.SphereLight):
            light.enable_virtual_sphere_shrinking = True
        if isinstance(light, (f2.PointLight, f2.SphereLight, f2.DiskLight, f2.RectLight)):
            light.enable_shaping = True
            light.shaping_cone_angle = 180.0
            light.shaping_focus = 2.0

    pixels = np.ones((8, 16, 3), dtype=np.float32)
    pixels[:, 8:, :] = [4.0, 2.0, 1.0]
    environment_path = tmp_path / "environment.exr"
    spy.Bitmap(np.zeros_like(pixels) if black_environment else pixels, srgb_gamma=False).write(
        environment_path
    )
    environment = scene.create_entity().create_component(f2.EnvMapLight)
    environment.env_map_path = environment_path

    emission_path = tmp_path / "emission.exr"
    spy.Bitmap(pixels, srgb_gamma=False).write(emission_path)
    material = scene.create_material(f2.StandardMaterial)
    material.emissive_factor = spy.float3(2.0)
    material.emissive_texture_path = emission_path
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    geometry.set_mesh_data(
        positions=np.array([[-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0]], dtype=np.float32),
        sub_mesh_indices=[np.array([[0, 2, 1], [2, 3, 1]], dtype=np.uint32)],
        normals=np.tile(np.array([0, 0, -1], dtype=np.float32), (4, 1)),
        texcoords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
        name="textured_emitter",
    )
    instance = scene.create_entity().create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]
    scene.update()

    power_sampler = f2.PowerLightSampler()
    hierarchical_sampler = f2.HierarchicalLightSampler()
    hierarchical_sampler.max_triangle_count_per_leaf = 1
    power_sampler.update(scene)
    hierarchical_sampler.update(scene)
    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sample_ref.slang")
    module, requirements = helper.prepare(scene, source_module)

    def bind(cursor: Any) -> None:
        helper.bind_scene(cursor)
        cursor["power_sampler"] = power_sampler
        cursor["hierarchical_sampler"] = hierarchical_sampler

    def function(name: str) -> Any:
        return (
            module[name]
            .as_func()
            .type_conformances(requirements.type_conformances)
            .write(bind)
            .return_type(spy.Tensor)
        )

    return function("probe"), function("evaluate_reference"), scene


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_light_sample_reference_black_environment(
    device_type: spy.DeviceType, device: spy.Device, tmp_path: Path
) -> None:
    probe, evaluate, scene = _make_probes(device, tmp_path, black_environment=True)
    receiver = spy.float3(0.0, 0.0, -4.0)
    fields = spy.Tensor.from_numpy(device, np.arange(4, dtype=np.uint32))
    result = evaluate(fields, 0, 2, 7, spy.float2(0.5), receiver, spy.float3(0.0)).to_numpy()
    np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(result[1], 0.0)

    seeds = spy.Tensor.from_numpy(device, np.arange(128, dtype=np.uint32))
    result = probe(seeds, 0, 7, receiver, receiver).to_numpy()
    np.testing.assert_array_equal(result, 0.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_light_sample_reference_round_trip(
    device_type: spy.DeviceType, device: spy.Device, tmp_path: Path
) -> None:
    probe, evaluate, scene = _make_probes(device, tmp_path)
    seeds = spy.Tensor.from_numpy(device, np.arange(512, dtype=np.uint32))
    for kind, light_id in [(0, i) for i in range(8)] + [(i, 0) for i in range(1, 4)]:
        result = probe(
            seeds, kind, light_id, spy.float3(0.0, 0.0, -4.0), spy.float3(0.7, 0.2, -3.0)
        ).to_numpy()
        assert np.count_nonzero(result[:, 0] == 1.0) > 20, (kind, light_id)
        assert np.all(np.isfinite(result)), (kind, light_id)
        np.testing.assert_array_equal(result[:, 3], 0.0, err_msg=str((kind, light_id)))
        np.testing.assert_allclose(result[:, 1:3], 0.0, atol=3e-4, err_msg=str((kind, light_id)))

    # Check the packed size and representative octahedral seam/fold directions.
    fields = spy.Tensor.from_numpy(device, np.arange(5, dtype=np.uint32))
    for parameters, direction in (
        ((0.0, 0.0), (0.0, 0.0, -1.0)),
        ((0.875, 0.875), (1.0, 1.0, -2.0)),
    ):
        expected = np.array(direction, dtype=np.float32)
        expected /= np.linalg.norm(expected)
        for emitter_type, light_id in ((2, 5), (1, 1)):
            result = evaluate(
                fields,
                0,
                emitter_type,
                light_id,
                spy.float2(*parameters),
                spy.float3(0.0, 0.0, -4.0),
                spy.float3(0.0),
            ).to_numpy()
            assert result[0, 3] == 1.0
            assert result[4, 0] == 16.0
            actual = result[3, :3] if emitter_type == 2 else result[2, :3]
            if emitter_type == 1:
                actual = actual / np.linalg.norm(actual)
            np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_light_sample_reference_zero_support_and_material_emission(
    device_type: spy.DeviceType, device: spy.Device, tmp_path: Path
) -> None:
    _, evaluate, scene = _make_probes(device, tmp_path)
    fields = spy.Tensor.from_numpy(device, np.arange(4, dtype=np.uint32))
    receiver = spy.float3(0.0, 0.0, -4.0)
    result = evaluate(fields, 3, 3, 0, spy.float2(0.25, 0.25), receiver, spy.float3(0.0)).to_numpy()
    assert result[0, 3] == 1.0
    assert np.all(result[0, :3] > 0.0)
    assert result[1, 3] > 0.0
    np.testing.assert_allclose(result[2, :3], [-0.5, -0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(result[1, :3] * result[1, 3], result[0, :3], rtol=1e-6)

    unsupported = evaluate(
        fields, 3, 3, 0, spy.float2(0.25, 0.25), receiver, spy.float3(0.0, 0.0, -1.0)
    ).to_numpy()
    assert unsupported[0, 3] == 1.0
    np.testing.assert_allclose(unsupported[0, :3], result[0, :3], rtol=1e-6)
    np.testing.assert_array_equal(unsupported[1], 0.0)

    other_texel = evaluate(
        fields, 3, 3, 0, spy.float2(0.1, 0.8), receiver, spy.float3(0.0)
    ).to_numpy()
    assert other_texel[0, 0] > 2.0 * result[0, 0]
    back = evaluate(fields, 3, 3, 0, spy.float2(0.25, 0.25), -receiver, spy.float3(0.0)).to_numpy()
    np.testing.assert_array_equal(back[0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(back[1], 0.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_light_sample_reference_point_and_degenerate_connections(
    device_type: spy.DeviceType, device: spy.Device, tmp_path: Path
) -> None:
    _, evaluate, scene = _make_probes(device, tmp_path)
    fields = spy.Tensor.from_numpy(device, np.arange(4, dtype=np.uint32))
    receiver = spy.float3(0.0, 0.0, -4.0)
    near = evaluate(fields, 1, 1, 0, spy.float2(0.0), receiver, spy.float3(0.0)).to_numpy()
    far = evaluate(fields, 1, 1, 0, spy.float2(0.0), 2.0 * receiver, spy.float3(0.0)).to_numpy()
    assert near[0, 3] == far[0, 3] == 1.0
    assert near[1, 3] > 0.0
    np.testing.assert_allclose(far[0, :3], near[0, :3] / 4.0, rtol=1e-6)
    np.testing.assert_allclose(far[1, 3], near[1, 3], rtol=1e-6)
    np.testing.assert_allclose(near[2], [0.0, 0.0, 0.0, 4.0], atol=1e-6)
    np.testing.assert_allclose(far[2], [0.0, 0.0, 0.0, 8.0], atol=1e-6)
    # Valid endpoints can coincide with the new receiver, making the connection undefined.
    for emitter_type, emitter_id, parameters, endpoint in (
        (1, 0, spy.float2(0.0), spy.float3(0.0)),
        (1, 2, spy.float2(0.0), spy.float3(0.0)),
        (3, 0, spy.float2(0.25), spy.float3(-0.5, -0.5, 0.0)),
    ):
        result = evaluate(
            fields, 1, emitter_type, emitter_id, parameters, endpoint, spy.float3(0.0)
        ).to_numpy()
        assert result[0, 3] == 0.0
        assert np.all(np.isfinite(result))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_light_sample_reference_virtual_proxy(
    device_type: spy.DeviceType, device: spy.Device, tmp_path: Path
) -> None:
    probe, evaluate, scene = _make_probes(device, tmp_path)
    seeds = spy.Tensor.from_numpy(device, np.arange(32, dtype=np.uint32))
    result = probe(seeds, 0, 1, spy.float3(0.0, 0.0, -0.1), spy.float3(0.0, 0.0, -4.0)).to_numpy()
    np.testing.assert_array_equal(result[:, 0], -1.0)
    np.testing.assert_allclose(result[:, 1], 0.0, rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(result[:, 2:], 0.0)
    fields = spy.Tensor.from_numpy(device, np.arange(4, dtype=np.uint32))
    result = evaluate(
        fields, 0, 1, 1, spy.float2(1.0), spy.float3(0.0, 0.0, -0.1), spy.float3(0.0)
    ).to_numpy()
    assert result[0, 3] == 0.0
