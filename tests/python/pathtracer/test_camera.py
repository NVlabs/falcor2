# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers


def matrix_to_numpy(matrix: spy.float4x4) -> np.ndarray:
    return np.array(
        [[float(matrix[row, col]) for col in range(4)] for row in range(4)], dtype=np.float32
    )


def fov_y_from_focal_length(focal_length: float, sensor_height: float) -> float:
    return float(np.degrees(2.0 * np.arctan(sensor_height / (2.0 * focal_length))))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_camera_modify(device_type: spy.DeviceType, device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    cam = helpers.create_test_camera(scene, width=1024, height=1024, fov_y=45.0)
    cam.fov_y = 90.0
    uniforms = cam.calc_uniforms()
    u = np.array(uniforms.image_u)
    v = np.array(uniforms.image_v)
    assert abs(np.dot(u, v)) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_camera_physical_projection_properties(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    cam = scene.create_entity().create_component(f2.Camera)

    cam.focal_length = 35.0
    assert cam.fov_y == pytest.approx(fov_y_from_focal_length(35.0, 24.0))

    cam.fov_y = 45.0
    expected_focal_length = 24.0 / (2.0 * np.tan(np.radians(45.0) * 0.5))
    assert cam.focal_length == pytest.approx(expected_focal_length)

    cam.sensor_height = 36.0
    assert cam.focal_length == pytest.approx(expected_focal_length)
    assert cam.fov_y == pytest.approx(fov_y_from_focal_length(expected_focal_length, 36.0))

    uniforms = cam.calc_uniforms()
    assert np.linalg.norm(np.array(uniforms.image_v)) == pytest.approx(
        cam.sensor_height / (2.0 * cam.focal_length)
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_camera_matrix_calculators(device_type: spy.DeviceType, device: spy.Device) -> None:
    """Camera matrices are calculated on demand, not GPU camera uniforms."""
    scene = f2.Scene.create(device)
    cam = helpers.create_test_camera(scene, width=64, height=64, fov_y=90.0)
    cam.depth_range = spy.float2(0.25, 250.0)
    uniforms = cam.calc_uniforms()

    assert not hasattr(uniforms, "view_from_world")
    assert not hasattr(uniforms, "clip_from_view")
    assert not hasattr(cam, "view_from_world")
    assert not hasattr(cam, "clip_from_view")
    assert hasattr(cam, "calc_view_from_world")
    assert hasattr(cam, "calc_clip_from_view")

    view = matrix_to_numpy(cam.calc_view_from_world())
    clip = matrix_to_numpy(cam.calc_clip_from_view())

    np.testing.assert_allclose(view, np.eye(4, dtype=np.float32), atol=1e-6)

    center_view = np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32)
    right_edge_view = np.array([1.0, 0.0, -1.0, 1.0], dtype=np.float32)
    top_edge_view = np.array([0.0, 1.0, -1.0, 1.0], dtype=np.float32)
    near_view = np.array([0.0, 0.0, -0.25, 1.0], dtype=np.float32)
    far_view = np.array([0.0, 0.0, -250.0, 1.0], dtype=np.float32)

    center_clip = clip @ center_view
    right_clip = clip @ right_edge_view
    top_clip = clip @ top_edge_view
    near_clip = clip @ near_view
    far_clip = clip @ far_view

    center_ndc = center_clip[:3] / center_clip[3]
    right_ndc = right_clip[:3] / right_clip[3]
    top_ndc = top_clip[:3] / top_clip[3]
    near_ndc = near_clip[:3] / near_clip[3]
    far_ndc = far_clip[:3] / far_clip[3]

    np.testing.assert_allclose(center_ndc[:2], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(right_ndc[0], 1.0, atol=1e-5)
    np.testing.assert_allclose(top_ndc[1], 1.0, atol=1e-5)
    np.testing.assert_allclose(near_ndc[2], 0.0, atol=1e-5)
    np.testing.assert_allclose(far_ndc[2], 1.0, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_camera_clip_uses_render_dims_without_dirtying(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Projection matrices can use render dimensions without changing camera state."""
    scene = f2.Scene.create(device)
    cam = helpers.create_test_camera(scene, width=1280, height=720, fov_y=90.0)
    scene.update()
    update_generation = scene.update_generation

    clip = matrix_to_numpy(cam.calc_clip_from_view(640, 480))
    right_edge_view = np.array([4.0 / 3.0, 0.0, -1.0, 1.0], dtype=np.float32)
    right_clip = clip @ right_edge_view
    right_ndc = right_clip[:3] / right_clip[3]

    np.testing.assert_allclose(right_ndc[0], 1.0, atol=1e-5)
    assert cam.width == 1280
    assert cam.height == 720
    assert scene.update_generation == update_generation


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_camera_uniforms_use_render_dims_without_dirtying(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    camera = helpers.create_test_camera(scene, width=1280, height=720)
    scene.update()
    update_generation = scene.update_generation

    render_camera = camera.calc_uniforms(640, 360)

    assert isinstance(render_camera, f2.CameraUniforms)
    assert render_camera.width == 640
    assert render_camera.height == 360
    assert render_camera.dims == spy.uint2(640, 360)
    assert render_camera == camera.calc_uniforms(640, 360)
    assert render_camera != camera.calc_uniforms(641, 360)
    np.testing.assert_allclose(
        np.linalg.norm(np.array(render_camera.image_u)),
        np.linalg.norm(np.array(render_camera.image_v)) * (640.0 / 360.0),
        rtol=1e-5,
    )
    assert camera.width == 1280
    assert camera.height == 720
    assert scene.update_generation == update_generation


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_camera_transform(device_type: spy.DeviceType, device: spy.Device) -> None:
    """Camera transform position/rotation affects uniforms."""
    scene = f2.Scene.create(device)
    cam = helpers.create_test_camera(scene, width=512, height=512)
    entity = cam.entity
    t = f2.Transform()
    t.translation = spy.float3(1.0, 2.0, 3.0)
    entity.transform = t
    uniforms = cam.calc_uniforms()
    pos = uniforms.position
    assert abs(pos[0] - 1.0) < 1e-5
    assert abs(pos[1] - 2.0) < 1e-5
    assert abs(pos[2] - 3.0) < 1e-5
