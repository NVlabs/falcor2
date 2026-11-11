# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
from falcor2.pxr import Usd, UsdGeom
from falcor2.rendernodes import ReferencePathTracerNode
from falcor2.testing import helpers
from tests.python.importers.mesh_motion_fixtures import (
    seam_stage,
    mesh_motion_options,
    add_motion_curve,
)
from tests.python.importers.mesh_motion_sampling import MeshSampler, sample_track
from tests.python.render.test_usd_ribbons import _make_probe


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_mesh_motion_playback(device: spy.Device) -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(2)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateNormalsAttr([(0, 0, 1)] * 3)
    mesh.SetNormalsInterpolation("vertex")
    curve = UsdGeom.BasisCurves.Define(stage, "/Curve")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([2])
    curve.SetWidthsInterpolation("constant")
    ribbon = UsdGeom.BasisCurves.Define(stage, "/Ribbon")
    ribbon.CreateTypeAttr("linear")
    ribbon.CreateCurveVertexCountsAttr([2])
    # Authored normals select the ribbon-to-mesh import path.
    ribbon.CreateNormalsAttr([(0, 0, 1)])
    ribbon.SetNormalsInterpolation("constant")
    ribbon.CreateWidthsAttr([1])
    ribbon.SetWidthsInterpolation("constant")
    for time in (0, 1):
        mesh.CreatePointsAttr().Set(
            [(-3, -1, 2 * time), (-1, -1, 2 * time), (-2, 1, 2 * time)], 2 * time
        )
        curve.CreatePointsAttr().Set([(2, -1, 2 * time), (2, 1, 2 * time)], 2 * time)
        curve.CreateWidthsAttr().Set([0.5 + 0.5 * time], 2 * time)
        ribbon.CreatePointsAttr().Set([(4, 0, 2 * time), (6, 0, 2 * time)], 2 * time)
    options = f2.ImportOptions()
    options.import_mesh_motion = True
    options.import_curve_motion = True
    imported = f2.UsdImporter().import_stage(stage, options)
    scene = f2.Scene.from_importer_scene(device, imported)
    assert any(isinstance(g, f2.DynamicMeshGeometry) for g in scene.geometries)
    assert any(isinstance(g, f2.DynamicCurveGeometry) for g in scene.geometries)
    assert scene.has_animation()
    scene.update()
    probe = _make_probe(device, scene)
    origins = np.array([[-2, 0, 4], [2, 0, 4], [5, 0.25, 4], [0, 0, 4]], dtype=np.float32)
    directions = np.tile(np.array([0, 0, -1], dtype=np.float32), (len(origins), 1))
    for time in (0, 0.5, 1):
        scene.time = time
        scene.update()
        hits = probe(origins, directions)
        assert [hit["valid"] for hit in hits] == [True, True, True, False]
        np.testing.assert_allclose(hits[0]["position"], [-2, 0, 2 * time], atol=1e-5)
        # The front of the curve moves with both its center and its interpolated radius.
        radius = (0.5 + 0.5 * time) / 2
        np.testing.assert_allclose(hits[1]["position"], [2, 0, 2 * time + radius], atol=1e-5)
        np.testing.assert_allclose(hits[2]["position"], [5, 0.25, 2 * time], atol=1e-5)
        for hit in hits[:3]:
            np.testing.assert_allclose(hit["normal"], [0, 0, 1], atol=1e-5)


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mesh_motion_playback_seeks(device: spy.Device) -> None:
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
    assert any(isinstance(g, f2.DynamicMeshGeometry) for g in scene.geometries)
    assert any(isinstance(g, f2.DynamicCurveGeometry) for g in scene.geometries)
    assert scene.has_animation()
    curve = imported.curves[0]
    curve_times = curve.motion_times
    curve_positions = np.array([curve.position_sample(i) for i in range(curve.sample_count)])
    curve_radii = np.array([curve.radius_sample(i) for i in range(curve.sample_count)])
    curve_indices = np.asarray(curve.indices, dtype=np.uint32)
    sampler = MeshSampler(mesh)
    indices = mesh.subgeometries[0].indices_numpy.copy()
    # Build the independent CPU reference using the existing static submission path.
    mesh.motion_times = []
    curve.motion_times = []
    reference = f2.Scene.from_importer_scene(device, imported)
    geometry = next(g for g in reference.geometries if isinstance(g, f2.StaticMeshGeometry))
    curve_geometry = next(g for g in reference.geometries if isinstance(g, f2.StaticCurveGeometry))
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

    def make_renderer(target: f2.Scene):
        camera = target.create_entity().create_component(f2.Camera)
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
        return camera, renderer

    camera, renderer = make_renderer(scene)
    reference_camera, reference_renderer = make_renderer(reference)
    times = (-1, 0.5, 1, 1, 1, 0.25, 0.25)
    for time in times:
        sampled = sampler.sample(time)
        geometry.set_mesh_data(sub_mesh_indices=[indices], texcoords=uv, **sampled)
        positions = sample_track(
            curve_times, curve_positions, curve.motion_interpolation_mode, time
        )
        radii = sample_track(curve_times, curve_radii, curve.motion_interpolation_mode, time)
        curve_geometry.set_curve_data(positions, radii, curve_indices)
        scene.time = time
        scene.update()
        reference.update()
        output, _ = renderer(scene, camera)
        expected, _ = reference_renderer(reference, reference_camera)
        pixels = output.to_numpy()
        assert np.isfinite(pixels).all()
        np.testing.assert_allclose(pixels, expected.to_numpy(), atol=1e-4)
        assert pixels[..., :3].min() < 0.1
        assert pixels[..., :3].max() > 0.1
