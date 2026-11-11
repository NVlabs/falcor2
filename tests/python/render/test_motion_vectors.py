# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_motion_vectors(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    scene.update()
    module = spy.Module.load_from_file(
        device, "render/test_motion_vectors.slang", link=scene.requirements.modules
    )

    def buffer(data: np.ndarray, stride: int) -> spy.Buffer:
        return device.create_buffer(
            usage=spy.BufferUsage.shader_resource, struct_size=stride, data=np.ravel(data)
        )

    # Nonzero vertex/index offsets and primitive index exercise the complete lookup key.
    triangle_vertices = np.zeros((7, 8), dtype=np.float32)
    triangle_vertices[4:7, :3] = [[0, 0, 2], [2, 0, 2], [0, 2, 2]]
    lss_vertices = np.zeros((5, 6), dtype=np.float32)
    lss_vertices[3:5, :4] = [[0, 0, 0, 0.5], [0, 0, 2, 0.5]]
    # Triangle uses indices 4..6; curve uses index 2. Both have index_offset=1.
    indices = np.array([0, 0, 2, 0, 3, 4, 5], dtype=np.uint32)
    previous_triangle_vertices = triangle_vertices.copy()
    previous_triangle_vertices[4:7, :3] = [[1, 0, 3], [4, 0, 3], [1, 4, 5]]
    previous_lss_vertices = lss_vertices.copy()
    previous_lss_vertices[3:5, :4] = [[1, -1, 0.5, 0.75], [1, -1, 2.5, 0.75]]
    index_buffer = buffer(indices, 4)
    # Both current vertex slots are nonzero; history is adjacent and indices use a separate slot.
    geometry_buffers = [
        index_buffer,
        index_buffer,
        buffer(triangle_vertices, 32),
        buffer(previous_triangle_vertices, 32),
        index_buffer,
        buffer(lss_vertices, 24),
        buffer(previous_lss_vertices, 24),
    ]
    resources = {
        "geometry": buffer(
            np.array(
                [f2.to_handle(resource).data for resource in geometry_buffers], dtype=np.uint64
            ),
            8,
        ),
    }
    current_world = np.array(
        [[0, -2, 0, 3], [3, 0, 0, -1], [0, 0, 0.5, 4], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    previous_world = np.array(
        [[1, 0, 0, -2], [0, 0, -1, 2], [0, 1, 0, 5], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    # The test device uses row-major matrix storage.
    resources["current_inverse"] = buffer(np.linalg.inv(current_world)[None], 64)
    resources["previous_world"] = buffer(previous_world[None], 64)

    def bind(cursor: spy.ShaderCursor) -> None:
        if cursor.has_field("g_scene"):
            scene.bind(cursor, f2.SceneBindFlags.all)
        for name, resource in resources.items():
            if cursor.has_field(name):
                cursor[name] = resource

    previous = module.previous_position.as_func().write(bind)

    for curve in (False, True):
        # Equal-radius +Z segment at axial midpoint and normalized angle 0.5 lies on +X.
        position_os = np.array([0.5, 0, 1] if curve else [0.4, 0.6, 2], dtype=np.float32)
        parameters = spy.float2(0.5, 0.5) if curve else spy.float2(0.2, 0.3)
        position_ws = (current_world @ np.append(position_os, 1))[:3]
        previous_position_os = (
            np.array([1.75, -1, 1.5], dtype=np.float32)
            if curve
            else np.array([0.5, 0.2, 0.3]) @ previous_triangle_vertices[4:7, :3]
        )
        for motion in (False, True):
            expected = (
                previous_world @ np.append(previous_position_os if motion else position_os, 1)
            )[:3]
            vertex_buffer = 5 if curve else 2
            # The first payload word packs uint16 vertex_buffer and uint16 index_buffer.
            descriptors = np.array(
                [[int(curve) | (int(motion) << 8), 0, 0, 0, vertex_buffer | (1 << 16), 1, 1, 0]],
                dtype=np.uint32,
            )
            resources["instance_descs"] = buffer(descriptors, 32)
            actual = previous(curve, 1, parameters, spy.float3(*position_ws))
            np.testing.assert_allclose(np.array(actual), expected, atol=2e-6)

            # Perturbing the supplied current point distinguishes flag-driven reconstruction
            # from the static fallback. History also differs from current vertices and radii.
            supplied = position_ws + np.array([0.125, -0.25, 0.5], dtype=np.float32)
            expected_branch = (
                expected
                if motion
                else (previous_world @ np.linalg.inv(current_world) @ np.append(supplied, 1))[:3]
            )
            actual = previous(curve, 1, parameters, spy.float3(*supplied))
            np.testing.assert_allclose(np.array(actual), expected_branch, atol=2e-6)

    # Other real surface types retain their rigid instance motion as well.
    resources["instance_descs"] = buffer(
        np.array([[2, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0]], dtype=np.uint32), 32
    )
    area_current = np.array([0.25, -0.75, 3.0], dtype=np.float32)
    expected_area = (previous_world @ np.linalg.inv(current_world) @ np.append(area_current, 1))[:3]
    actual_area = module.previous_area_position.as_func().write(bind)(spy.float3(*area_current))
    np.testing.assert_allclose(np.array(actual_area), expected_area, atol=2e-6)

    def project(position: np.ndarray, camera: np.ndarray) -> np.ndarray:
        x, y, z = position - camera
        return np.array([(x / z * 0.5 + 0.5) * 100, (0.5 - y / z * 0.5) * 80])

    current = np.array([0.125, -0.375, 3.25], dtype=np.float32)
    previous_position = np.array([-0.75, 0.25, 4.0])
    camera_position = np.array([0.2, -0.1, 0.0])
    expected = project(previous_position, camera_position) - project(current, np.zeros(3))
    actual = module.motion(
        spy.float3(*current), spy.float3(*previous_position), spy.float3(*camera_position)
    )
    np.testing.assert_allclose(np.array(actual), expected, atol=4e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_static_geometry_motion_flags(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    mesh = scene.create_geometry(f2.StaticMeshGeometry)
    mesh.set_mesh_data(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        sub_mesh_indices=[np.array([[0, 1, 2]], dtype=np.uint32)],
    )
    curve = scene.create_geometry(f2.StaticCurveGeometry)
    curve.set_curve_data(
        positions=np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32),
        radii=np.array([0.1, 0.1], dtype=np.float32),
        indices=np.array([0, 1], dtype=np.uint32),
    )
    instances = []
    material = scene.create_material(f2.StandardMaterial)
    for geometry in (mesh, curve):
        instance = scene.create_entity().create_component(f2.GeometryInstance)
        instance.geometry = geometry
        instance.materials = [material]
        instances.append(instance)
    scene.update()
    helper = f2.SceneShaderHelper(device)
    module, requirements = helper.prepare(scene, "render/test_motion_vectors.slang")
    probe = module.static_descriptor_flags.as_func().write(helper.bind_scene)
    for instance, count in zip(instances, (3, 1)):
        actual = probe(int(instance.geometry_instance_id))
        np.testing.assert_array_equal(np.array(actual), [count, 0])
