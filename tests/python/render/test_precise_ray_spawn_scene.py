# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.rendernodes import ReferencePathTracerNode


def _create_scene(
    device: spy.Device, baked: bool, resolution: int = 64
) -> tuple[f2.Scene, f2.Camera, ReferencePathTracerNode]:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial, f2.Properties({"double_sided": True}))
    # The plane ends up at world z=0. Interpolating its large local coordinates
    # loses more precision than the old offset based only on world position covers.
    positions = np.array(
        [[-500, -500, 24000], [500, -500, 24000], [-500, 500, 24000], [500, 500, 24000]],
        dtype=np.float32,
    )
    scale = np.array([0.002, 0.004, 0.125], dtype=np.float32)
    translation = np.array([0, 0, -3000], dtype=np.float32)
    if baked:
        positions = (positions.astype(np.float64) * scale + translation).astype(np.float32)
    mesh = scene.create_geometry(f2.StaticMeshGeometry)
    mesh.set_mesh_data(
        positions=positions,
        normals=np.tile(np.array([0, 0, 1], dtype=np.float32), (4, 1)),
        tangents=np.tile(np.array([1, 0, 0], dtype=np.float32), (4, 1)),
        sub_mesh_indices=[np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)],
    )
    entity = scene.create_entity()
    if not baked:
        transform = f2.Transform()
        transform.scale = spy.float3(*scale)
        transform.translation = spy.float3(*translation)
        entity.transform = transform
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = mesh
    instance.materials = [material]
    scene.create_entity().create_component(
        f2.EnvMapLight
    ).env_map_path = "data/assets/envmaps/aerodynamics_workshop_512.hdr"
    camera = helpers.create_test_camera(
        scene,
        width=resolution,
        height=resolution,
        position=spy.float3(0, 0, 3),
        rotation=spy.math.quat_from_look_at(spy.float3(0, 0, -1), spy.float3(0, 1, 0)),
    )
    scene.update()
    node = ReferencePathTracerNode.create(device)
    node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    node.max_depth = 3
    node.enable_nee = True
    return scene, camera, node


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_transformed_plane_matches_baked_geometry(device: spy.Device) -> None:

    # Generate 3 images -
    # - reference (i.e. baked transform),
    # - legacy (i.e. non-baked transform without precise ray offsets),
    # - precise (i.e. non-baked transform with precise ray offsets)
    scene, camera, node = _create_scene(device, baked=True)
    node.pixel_filter = f2.BoxReconstructionFilter(width=0)
    image, _ = node(scene, camera, iteration=0)
    reference = image.to_numpy()[..., :3].copy()

    scene, camera, node = _create_scene(device, baked=False)
    assert not node.enable_precise_ray_offsets
    node.pixel_filter = f2.BoxReconstructionFilter(width=0)
    image, _ = node(scene, camera, iteration=0)
    legacy = image.to_numpy()[..., :3].copy()

    node.enable_precise_ray_offsets = True
    node.pixel_filter = f2.BoxReconstructionFilter(width=0)
    image, _ = node(scene, camera, iteration=0)
    precise = image.to_numpy()[..., :3].copy()

    # Check that precise is close to reference
    assert np.isfinite(precise).all()
    np.testing.assert_allclose(precise, reference, atol=1e-5, rtol=1e-5)

    # And that legacy differs significantly from reference
    assert np.count_nonzero(np.max(np.abs(legacy - reference), axis=-1) > 0.01) > 100


def _render_small_gap(device: spy.Device, scale: float, precise: bool) -> np.ndarray:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial, f2.Properties({"double_sided": True}))
    # The camera sees the lower plane to the left of the raised half-plane.
    # A fixed world-space offset can jump above the occluder in the small copy.
    for positions in (
        [[-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0]],
        [[0, -1, 0.1], [1, -1, 0.1], [0, 1, 0.1], [1, 1, 0.1]],
    ):
        mesh = scene.create_geometry(f2.StaticMeshGeometry)
        mesh.set_mesh_data(
            positions=np.array(positions, dtype=np.float32) * scale,
            normals=np.tile(np.array([0, 0, 1], dtype=np.float32), (4, 1)),
            tangents=np.tile(np.array([1, 0, 0], dtype=np.float32), (4, 1)),
            sub_mesh_indices=[np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)],
        )
        instance = scene.create_entity().create_component(f2.GeometryInstance)
        instance.geometry = mesh
        instance.materials = [material]
    scene.create_entity().create_component(
        f2.EnvMapLight
    ).env_map_path = "data/assets/envmaps/aerodynamics_workshop_512.hdr"
    camera = helpers.create_test_camera(
        scene,
        width=64,
        height=64,
        position=spy.float3(0, 0, 3 * scale),
        rotation=spy.math.quat_from_look_at(spy.float3(0, 0, -1), spy.float3(0, 1, 0)),
    )
    scene.update()
    node = ReferencePathTracerNode.create(device)
    node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    node.max_depth = 3
    node.enable_precise_ray_offsets = precise
    node.pixel_filter = f2.BoxReconstructionFilter(width=0)
    image, _ = node(scene, camera, iteration=0)
    return image.to_numpy()[..., :3].copy()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_precise_offsets_preserve_small_gap_visibility(device: spy.Device) -> None:
    reference = _render_small_gap(device, 1.0, True)
    small = _render_small_gap(device, 0.0001, True)
    legacy = _render_small_gap(device, 0.0001, False)
    assert np.isfinite(small).all()
    np.testing.assert_allclose(small, reference, atol=1e-4, rtol=1e-4)
    assert np.count_nonzero(np.max(np.abs(legacy - reference), axis=-1) > 0.01) > 10
