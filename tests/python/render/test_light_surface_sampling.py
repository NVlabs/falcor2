# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampling and evaluation contract tests for finite ILight surfaces."""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
from falcor2.editor import SceneShaderHelper
import falcor2.testing.helpers as helpers
from falcor2.utils.chi2 import ChiSquareTest


SAMPLES_PER_THREAD = 1024
SPHERE_CENTER = spy.float3(-4.0, 0.0, 0.0)
DISK_CENTER = spy.float3(0.0, 0.0, 0.0)
RECT_CENTER = spy.float3(4.0, 0.0, 0.0)
CYLINDER_CENTER = spy.float3(8.0, 0.0, 0.0)
AUTHORED_SPHERE_CENTER = spy.float3(-8.0, 0.0, 0.0)
SPHERE_COLOR = spy.float3(1.0, 0.5, 0.25)
DISK_COLOR = spy.float3(0.5, 1.0, 0.25)
RECT_COLOR = spy.float3(0.25, 0.5, 1.0)
CYLINDER_COLOR = spy.float3(1.0, 0.25, 0.5)


def _resource_scope(fixture_name: str, config: pytest.Config) -> Literal["function", "module"]:
    """Select fixture lifetime from the configured device cache policy."""
    return "function" if config.getoption("--device-cache-policy") == "test" else "module"


@dataclass(frozen=True)
class _Resources:
    device: spy.Device
    scene: f2.Scene
    helper: SceneShaderHelper
    module: spy.Module


def _add_light(
    scene: f2.Scene,
    light_type: type[f2.Light],
    translation: spy.float3,
    scale: spy.float3,
) -> f2.Light:
    entity = scene.create_entity()
    transform = f2.Transform()
    transform.translation = translation
    transform.scale = scale
    entity.transform = transform
    return entity.create_component(light_type)


@pytest.fixture(scope=_resource_scope, params=helpers.DEFAULT_DEVICE_TYPES)
def _resources(request: pytest.FixtureRequest) -> _Resources:
    device_type = cast(spy.DeviceType, request.param)
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)

    point_entity = scene.create_entity()
    point_entity.create_component(f2.PointLight)

    sphere = cast(
        f2.SphereLight,
        _add_light(scene, f2.SphereLight, SPHERE_CENTER, spy.float3(2.0)),
    )
    sphere.radius = 1.0
    sphere.color = SPHERE_COLOR
    sphere.enable_virtual_sphere_shrinking = True
    sphere.enable_shaping = True
    sphere.shaping_cone_angle = 180.0
    sphere.shaping_focus = 2.0

    disk = cast(
        f2.DiskLight,
        _add_light(scene, f2.DiskLight, DISK_CENTER, spy.float3(2.0, 1.0, 1.0)),
    )
    disk.radius = 1.0
    disk.color = DISK_COLOR

    rect = cast(
        f2.RectLight,
        _add_light(scene, f2.RectLight, RECT_CENTER, spy.float3(2.0, 3.0, 1.0)),
    )
    rect.color = RECT_COLOR

    cylinder = cast(
        f2.CylinderLight,
        _add_light(scene, f2.CylinderLight, CYLINDER_CENTER, spy.float3(2.0, 1.0, 1.0)),
    )
    cylinder.radius = 1.0
    cylinder.color = CYLINDER_COLOR

    authored_sphere = cast(
        f2.SphereLight,
        _add_light(scene, f2.SphereLight, AUTHORED_SPHERE_CENTER, spy.float3(2.0)),
    )
    authored_sphere.radius = 1.0
    authored_sphere.color = SPHERE_COLOR

    scene.update()
    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/light_surface_sampling_tests.slang")
    module, _ = helper.prepare(scene, source_module)
    return _Resources(device=device, scene=scene, helper=helper, module=module)


def _surface_function(resources: _Resources, name: str) -> Any:
    return (
        resources.module[f"light_surface_sampling_tests::{name}"]
        .as_func()
        .type_conformances(resources.scene.requirements.type_conformances)
        .write(resources.helper.bind_scene)
    )


@pytest.mark.nightly_if_not_vk(device_param="_resources")
def test_light_surface_sampling_chi2(_resources: _Resources) -> None:
    """Verify analytic area-light samples follow their surface distributions."""
    configs = (
        (
            "light_surface_sampling_tests::AreaLightDirectionTarget",
            {
                "light_id": 1,
                "receiver_pos_ws": spy.float3(-4.0, 0.0, 3.0),
                "domain_min": spy.float2(-np.pi, 0.7),
                "domain_max": spy.float2(np.pi, 1.0),
            },
            spy.uint2(64, 64),
        ),
        (
            "light_surface_sampling_tests::AreaLightDirectionTarget",
            {
                "light_id": 2,
                "receiver_pos_ws": spy.float3(0.0, 0.0, -3.0),
                "domain_min": spy.float2(-np.pi, -1.0),
                "domain_max": spy.float2(np.pi, -0.8),
            },
            spy.uint2(64, 64),
        ),
        (
            "light_surface_sampling_tests::AreaLightDirectionTarget",
            {
                "light_id": 3,
                "receiver_pos_ws": spy.float3(4.0, 0.0, -3.0),
                "domain_min": spy.float2(-np.pi, -1.0),
                "domain_max": spy.float2(np.pi, -0.84),
            },
            spy.uint2(64, 64),
        ),
        (
            "light_surface_sampling_tests::CylinderLightSurfaceTarget",
            {
                "light_id": 4,
                "receiver_pos_ws": spy.float3(8.0, 0.0, -3.0),
                "center_ws": CYLINDER_CENTER,
                "axis_ws": spy.float3(1.0, 0.0, 0.0),
                "radius_u_ws": spy.float3(0.0, 1.0, 0.0),
                "radius_v_ws": spy.float3(0.0, 0.0, 1.0),
                "half_length_ws": 1.0,
            },
            spy.uint2(64, 64),
        ),
    )

    for target, bindings, resolution in configs:
        test = ChiSquareTest(
            device=_resources.device,
            module=_resources.module,
            target=target,
            target_bindings=bindings,
            global_bindings=_resources.helper.bind_scene,
            type_conformances=_resources.scene.requirements.type_conformances,
            res=resolution,
            sample_count=250000,
            ires=64,
        )
        assert test.run(test_count=len(configs)), test.messages


@pytest.mark.parametrize(
    "light_id,receiver_pos_ws,expect_valid,expect_invalid,expected_nee_only",
    (
        pytest.param(1, spy.float3(-4.0, 0.0, 5.0), True, False, False, id="sphere"),
        pytest.param(2, spy.float3(0.0, 0.0, -3.0), True, False, False, id="disk"),
        pytest.param(3, spy.float3(4.0, 0.0, -3.0), True, False, False, id="rectangle"),
        pytest.param(4, spy.float3(8.0, 0.0, -3.0), True, True, False, id="cylinder"),
        pytest.param(1, spy.float3(-4.0, 0.0, 2.002), True, False, True, id="sphere-near-surface"),
        pytest.param(5, AUTHORED_SPHERE_CENTER, False, True, False, id="sphere-inside"),
        pytest.param(5, spy.float3(-8.0, 0.0, 2.0), False, True, False, id="sphere-on-surface"),
    ),
)
def test_light_surface_eval_sample_consistency(
    _resources: _Resources,
    light_id: int,
    receiver_pos_ws: spy.float3,
    expect_valid: bool,
    expect_invalid: bool,
    expected_nee_only: bool,
) -> None:
    """Verify sampled surface data agrees with the new ILight evaluators."""
    sample_count = 262144
    thread_count = (sample_count + SAMPLES_PER_THREAD - 1) // SAMPLES_PER_THREAD
    pdf_errors = spy.Tensor.zeros(_resources.device, (thread_count,), dtype=float)
    normal_errors = spy.Tensor.zeros(_resources.device, (thread_count,), dtype=float)
    weight_errors = spy.Tensor.zeros(_resources.device, (thread_count,), dtype=float)
    geometry_errors = spy.Tensor.zeros(_resources.device, (thread_count,), dtype=float)
    counters = spy.Tensor.zeros(_resources.device, (3,), dtype="uint")

    kernel = _surface_function(_resources, "test_eval_sample_consistency")
    kernel(
        tid=spy.grid((thread_count,)),
        sample_count=sample_count,
        samples_per_thread=SAMPLES_PER_THREAD,
        seed=1234,
        light_id=light_id,
        receiver_pos_ws=receiver_pos_ws,
        expected_nee_only=expected_nee_only,
        pdf_errors=pdf_errors,
        normal_errors=normal_errors,
        weight_errors=weight_errors,
        geometry_errors=geometry_errors,
        counters=counters,
    )

    counts = counters.to_numpy()
    assert (counts[0] > 0) == expect_valid
    assert (counts[1] > 0) == expect_invalid
    assert counts[2] == 0
    assert np.max(pdf_errors.to_numpy()) < 2e-5
    assert np.max(normal_errors.to_numpy()) < 2e-6
    assert np.max(weight_errors.to_numpy()) < 2e-5
    assert np.max(geometry_errors.to_numpy()) < 2e-5


def test_light_surface_evaluation_contract(_resources: _Resources) -> None:
    """Verify defaults, sidedness, transforms, and analytic surface PDFs."""
    values = _surface_function(_resources, "eval_surface_values")
    metadata = _surface_function(_resources, "eval_surface_metadata")
    sample_flags = _surface_function(_resources, "sample_flags")

    light_ids = spy.Tensor.from_numpy(_resources.device, np.arange(5, dtype=np.uint32))
    receiver_positions = spy.Tensor.from_numpy(
        _resources.device,
        np.array(
            [
                [0.0, 0.0, 3.0],
                [-4.0, 0.0, 5.0],
                [0.0, 0.0, -3.0],
                [4.0, 0.0, -3.0],
                [8.0, 0.0, -3.0],
            ],
            dtype=np.float32,
        ),
    )
    emitter_positions = spy.Tensor.from_numpy(
        _resources.device,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [-4.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [8.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        ),
    )
    emitter_normals = spy.Tensor.from_numpy(
        _resources.device,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        ),
    )

    evaluated = values(light_ids, receiver_positions, emitter_positions, emitter_normals).to_numpy()
    sphere_cos_cone_angle = np.sqrt(1.0 - 4.0 / 25.0)
    sphere_pdf = 1.0 / (2.0 * np.pi * (1.0 - sphere_cos_cone_angle))
    expected_pdfs = np.array(
        [0.0, sphere_pdf, 9.0 / (2.0 * np.pi), 1.5, 1.0 / np.pi], dtype=np.float32
    )
    expected_radiance = np.vstack(
        [np.zeros(3, dtype=np.float32), SPHERE_COLOR, DISK_COLOR, RECT_COLOR, CYLINDER_COLOR]
    )
    np.testing.assert_allclose(evaluated[:, 0], expected_pdfs, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(evaluated[:, 1:], expected_radiance, rtol=1e-6, atol=0.0)

    flag_light_ids = spy.Tensor.from_numpy(
        _resources.device, np.array([0, 1, 2, 3, 1], dtype=np.uint32)
    )
    flag_receiver_positions = spy.Tensor.from_numpy(
        _resources.device,
        np.array(
            [
                [0.0, 0.0, 3.0],
                [-4.0, 0.0, 5.0],
                [0.0, 0.0, -3.0],
                [4.0, 0.0, -3.0],
                [-4.0, 0.0, 2.002],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        sample_flags(flag_light_ids, flag_receiver_positions).to_numpy(),
        # nee_only; has_surface_point; has_surface_point; has_surface_point; both.
        np.array([2, 1, 1, 1, 3], dtype=np.uint32),
    )

    evaluated_metadata = metadata(light_ids, receiver_positions, emitter_positions).to_numpy()
    np.testing.assert_allclose(
        evaluated_metadata,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, -1.0, 1.0],
                [0.0, 0.0, -1.0, 1.0],
                [0.0, 0.0, -1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=0.0,
    )

    back_receivers = spy.Tensor.from_numpy(
        _resources.device,
        np.array(
            [
                [0.0, 0.0, -3.0],
                [-4.0, 0.0, -5.0],
                [0.0, 0.0, 3.0],
                [4.0, 0.0, 3.0],
                [8.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        ),
    )
    back_values = values(light_ids, back_receivers, emitter_positions, emitter_normals).to_numpy()
    np.testing.assert_allclose(back_values, 0.0, atol=0.0)
    back_metadata = metadata(light_ids, back_receivers, emitter_positions).to_numpy()
    np.testing.assert_array_equal(back_metadata[:, 3], [0.0, 0.0, 0.0, 0.0, 0.0])

    near_receiver = spy.Tensor.from_numpy(
        _resources.device, np.array([[-4.0, 0.0, 2.002]], dtype=np.float32)
    )
    sphere_id = spy.Tensor.from_numpy(_resources.device, np.array([1], dtype=np.uint32))
    sphere_emitter = spy.Tensor.from_numpy(
        _resources.device, np.array([[-4.0, 0.0, 2.0]], dtype=np.float32)
    )
    near_metadata = metadata(sphere_id, near_receiver, sphere_emitter).to_numpy()
    near_values = values(
        sphere_id,
        near_receiver,
        sphere_emitter,
        spy.Tensor.from_numpy(_resources.device, np.array([[0.0, 0.0, 1.0]], dtype=np.float32)),
    ).to_numpy()
    assert near_metadata[0, 3] == 0.0
    np.testing.assert_array_equal(near_values, 0.0)

    intersect = _surface_function(_resources, "eval_surface_intersection")
    origins = spy.Tensor.from_numpy(
        _resources.device,
        np.array(
            [
                [0.0, 0.0, 3.0],
                [-4.0, 0.0, 5.0],
                [0.0, 0.0, -3.0],
                [4.0, 0.0, -3.0],
                [8.0, 0.0, -3.0],
            ],
            dtype=np.float32,
        ),
    )
    directions = spy.Tensor.from_numpy(
        _resources.device,
        np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    intersections = intersect(light_ids, origins, directions, 0.0, 10.0).to_numpy()
    np.testing.assert_allclose(
        intersections, np.array([-1.0, 3.0, 3.0, 3.0, 2.0], dtype=np.float32)
    )

    cylinder_id = spy.Tensor.from_numpy(_resources.device, np.array([4], dtype=np.uint32))
    cylinder_center = spy.Tensor.from_numpy(
        _resources.device, np.array([[8.0, 0.0, 0.0]], dtype=np.float32)
    )
    cylinder_axis = spy.Tensor.from_numpy(
        _resources.device, np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        intersect(cylinder_id, cylinder_center, cylinder_axis, 0.0, 10.0).to_numpy(), -1.0
    )

    close_sphere_intersection = intersect(
        sphere_id,
        near_receiver,
        spy.Tensor.from_numpy(_resources.device, np.array([[0.0, 0.0, -1.0]], dtype=np.float32)),
        0.0,
        10.0,
    ).to_numpy()
    np.testing.assert_array_equal(close_sphere_intersection, -1.0)

    authored_sphere_id = spy.Tensor.from_numpy(_resources.device, np.array([5], dtype=np.uint32))
    authored_sphere_receiver = spy.Tensor.from_numpy(
        _resources.device, np.array([[-8.0, 0.0, 2.002]], dtype=np.float32)
    )
    authored_sphere_emitter = spy.Tensor.from_numpy(
        _resources.device, np.array([[-8.0, 0.0, 2.0]], dtype=np.float32)
    )
    authored_sphere_values = values(
        authored_sphere_id,
        authored_sphere_receiver,
        authored_sphere_emitter,
        spy.Tensor.from_numpy(_resources.device, np.array([[0.0, 0.0, 1.0]], dtype=np.float32)),
    ).to_numpy()
    assert authored_sphere_values[0, 0] > 0.0
    np.testing.assert_allclose(authored_sphere_values[0, 1:], SPHERE_COLOR, rtol=1e-6, atol=0.0)

    authored_sphere_intersection = intersect(
        authored_sphere_id,
        authored_sphere_receiver,
        spy.Tensor.from_numpy(_resources.device, np.array([[0.0, 0.0, -1.0]], dtype=np.float32)),
        0.0,
        10.0,
    ).to_numpy()
    np.testing.assert_allclose(authored_sphere_intersection, 0.002, rtol=1e-4, atol=1e-6)

    clipped_intersections = intersect(light_ids, origins, directions, 0.0, 1.9).to_numpy()
    np.testing.assert_allclose(clipped_intersections, -1.0)
