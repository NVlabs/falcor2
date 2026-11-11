# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
from falcor2.pxr import UsdGeom
from falcor2.rendernodes import ReferencePathTracerNode
from falcor2.testing import helpers
from tests.python.importers.mesh_motion_fixtures import (
    seam_stage,
    mesh_motion_options,
    add_motion_curve,
)
from tests.python.importers.mesh_motion_sampling import MeshSampler, sample_track


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_mesh_motion_static_submission(device: spy.Device) -> None:
    stage = seam_stage()
    add_motion_curve(stage)
    source = UsdGeom.Mesh(stage.GetPrimAtPath("/Seam"))
    default_points = np.asarray(source.GetPointsAttr().Get(0)).copy()
    default_points[:, 2] = -1
    source.GetPointsAttr().Set(default_points.tolist())
    assert not f2.ImportOptions().import_mesh_motion
    assert not f2.ImportOptions().import_curve_motion
    ordinary = f2.UsdImporter().import_stage(stage)
    assert not ordinary.meshes[0].motion_times
    assert not ordinary.curves[0].motion_times
    assert ordinary.curves[0].sample_count == 1
    np.testing.assert_allclose(ordinary.meshes[0].positions[:, 2], -1)

    options = mesh_motion_options()
    options.import_curve_motion = True
    imported = f2.UsdImporter().import_stage(stage, options)
    mesh = imported.meshes[0]
    assert mesh.motion_times == [0, 1]
    scene = f2.Scene.from_importer_scene(device, imported)
    geometry = next(g for g in scene.geometries if isinstance(g, f2.StaticMeshGeometry))
    curve = imported.curves[0]
    curve_geometry = next(g for g in scene.geometries if isinstance(g, f2.StaticCurveGeometry))
    curve_indices = np.asarray(curve.indices, dtype=np.uint32)
    curve_positions = np.array([curve.position_sample(i) for i in range(curve.sample_count)])
    curve_radii = np.array([curve.radius_sample(i) for i in range(curve.sample_count)])
    np.testing.assert_allclose(
        curve_geometry.local_aabb.min,
        (curve.positions - curve.radii[:, None]).min(axis=0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        curve_geometry.local_aabb.max,
        (curve.positions + curve.radii[:, None]).max(axis=0),
        atol=1e-6,
    )
    indices = mesh.subgeometries[0].indices_numpy.copy()
    np.testing.assert_allclose(geometry.positions(0)[geometry.indices(0)], mesh.positions[indices])
    # A nonzero static attribute index must survive the scene bridge's motion filter.
    uv = mesh.texcoords(0).copy()
    extra = f2.ImporterMesh.create(
        [indices], {"position": mesh.positions.copy(), "tex_coord_1": uv}
    )
    extra_scene = f2.ImporterScene()
    extra_scene.meshes = [extra]
    node = f2.ImporterNode()
    node.mesh_index = 0
    extra_scene.nodes = [node]
    loaded = f2.Scene.from_importer_scene(device, extra_scene)
    extra_geometry = next(g for g in loaded.geometries if isinstance(g, f2.StaticMeshGeometry))
    assert any(
        a.index == 1 and a.semantic == f2.MeshAttributeSemantic.tex_coord
        for a in extra_geometry.vertex_attributes(0)
    )

    camera = scene.create_entity().create_component(f2.Camera)
    camera.width = camera.height = 32
    transform = f2.Transform()
    transform.translation = spy.float3(0, 0, 5)
    transform.rotation = spy.math.quat_from_look_at(spy.float3(0, 0, -1), spy.float3(0, 1, 0))
    camera.entity.transform = transform
    renderer = ReferencePathTracerNode.create(device)
    renderer.max_depth = 1
    renderer.use_background_color = True
    renderer.background_color = spy.float3(0.2)
    renderer.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float, (32, 32))

    sampler = MeshSampler(mesh)
    for time in (0, 0.5, 1):
        sampled = sampler.sample(time)
        geometry.set_mesh_data(sub_mesh_indices=[indices], texcoords=uv, **sampled)
        positions = sample_track(
            curve.motion_times, curve_positions, curve.motion_interpolation_mode, time
        )
        radii = sample_track(curve.motion_times, curve_radii, curve.motion_interpolation_mode, time)
        curve_geometry.set_curve_data(positions, radii, curve_indices)
        scene.update()
        np.testing.assert_allclose(
            geometry.positions(0)[geometry.indices(0)], sampled["positions"][indices]
        )
        np.testing.assert_allclose(
            curve_geometry.local_aabb.min, (positions - radii[:, None]).min(axis=0), atol=1e-6
        )
        np.testing.assert_allclose(
            curve_geometry.local_aabb.max, (positions + radii[:, None]).max(axis=0), atol=1e-6
        )
        output, _ = renderer(scene, camera)
        pixels = output.to_numpy()
        assert np.isfinite(pixels).all()
        assert pixels[..., :3].min() < 0.1
        assert pixels[..., :3].max() > 0.1
