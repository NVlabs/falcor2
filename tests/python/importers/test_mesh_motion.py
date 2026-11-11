# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import numpy as np
import pytest
import falcor2 as f2
from falcor2.pxr import Usd, UsdGeom, Sdf
from .mesh_motion_fixtures import seam_stage, mesh_motion_options
from .mesh_motion_sampling import MeshSampler

pytestmark = pytest.mark.nightly

ASSETS = Path(__file__).resolve().parents[3] / "data/assets/test_mesh_motion"
P, N, T = (f2.ImporterSemantic.position, f2.ImporterSemantic.normal, f2.ImporterSemantic.tangent)


def test_indexed_stream_construction() -> None:
    positions = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    mesh = f2.ImporterMesh.create([indices], {"position_1": positions + 1, "position": positions})
    np.testing.assert_array_equal(mesh.positions, positions)
    np.testing.assert_array_equal(mesh.find_stream(P, 1), positions + 1)
    with pytest.raises(RuntimeError, match="already set"):
        f2.ImporterMesh.create(
            [indices], {"position": positions, "position_1": positions, "position_01": positions}
        )
    with pytest.raises(RuntimeError, match="vertex count"):
        f2.ImporterMesh.create([indices], {"position": positions, "position_1": positions[:2]})


def _frames(mesh: f2.ImporterMesh, semantic: f2.ImporterSemantic) -> np.ndarray:
    return np.array([mesh.find_stream(semantic, i) for i in range(len(mesh.motion_times))])


@pytest.mark.parametrize("recompute", [False, True])
def test_usd_samples_and_tangent_transfer(recompute: bool) -> None:
    stage = seam_stage()
    options = mesh_motion_options()
    options.recompute_normals = recompute
    options.mesh_refinement_level = 3
    imported = f2.UsdImporter().import_stage(stage, options)
    mesh = imported.meshes[0]
    assert mesh.properties["refinement_level"] == 3
    p, n, t = [_frames(mesh, semantic) for semantic in (P, N, T)]
    np.testing.assert_array_equal(
        p, [UsdGeom.Mesh(stage.GetPrimAtPath("/Seam")).GetPointsAttr().Get(time) for time in (0, 1)]
    )
    np.testing.assert_allclose(mesh.positions, p[0])
    np.testing.assert_allclose(mesh.normals, n[0])
    np.testing.assert_allclose(mesh.tangents, t[0], atol=1e-6)
    expected = t[0][:, [1, 0, 2]].copy()
    expected[:, 0] *= -1
    np.testing.assert_allclose(t[1], expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(t, axis=-1), 1, atol=1e-6)
    np.testing.assert_allclose(np.sum(n * t, axis=-1), 0, atol=1e-6)
    assert set(mesh.handedness.ravel()) == ({1.0} if recompute else {-1.0, 1.0})
    snapshot = mesh.find_stream(P, 1).copy()
    snapshot[:] = 99
    np.testing.assert_array_equal(_frames(mesh, P), p)
    sampler = MeshSampler(mesh)
    for time in (1, 0.5, 0, -1, 2):
        w = np.clip(time, 0, 1)
        np.testing.assert_allclose(sampler.sample(time)["positions"], (1 - w) * p[0] + w * p[1])


def test_mesh_motion_nonpositive_time_rate() -> None:
    stage = seam_stage()
    for rate in (0, -24):
        stage.SetTimeCodesPerSecond(rate)
        mesh = f2.UsdImporter().import_stage(stage, mesh_motion_options()).meshes[0]
        np.testing.assert_allclose(mesh.motion_times, [0, 1 / 24])


def test_usd_union_schedule_and_fixed_layout() -> None:
    stage = seam_stage()
    source = UsdGeom.Mesh(stage.GetPrimAtPath("/Seam"))
    source.GetNormalsAttr().Set([(0, 0, 1)] * 6, 0.25)
    source.GetNormalsAttr().Set([(0, 0.6, 0.8)] * 6, 0.75)
    stage.SetInterpolationType(Usd.InterpolationTypeHeld)
    mesh = f2.UsdImporter().import_stage(stage, mesh_motion_options()).meshes[0]
    assert mesh.motion_times == [0, 0.25, 0.75, 1]
    assert mesh.motion_interpolation_mode == f2.AnimationInterpolationMode.step
    np.testing.assert_allclose(
        _frames(mesh, P), [source.GetPointsAttr().Get(t) for t in mesh.motion_times]
    )
    np.testing.assert_allclose(
        _frames(mesh, N),
        [source.GetNormalsAttr().Get(t) for t in mesh.motion_times],
        atol=1e-6,
    )
    source.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3, 4, 5], 0)
    source.GetFaceVertexIndicesAttr().Set([1, 0, 2, 3, 4, 5], 0.5)
    with pytest.raises(RuntimeError, match="changes indices"):
        f2.UsdImporter().import_stage(stage, mesh_motion_options())
    source.GetFaceVertexIndicesAttr().ClearAtTime(0.5)
    source.GetFaceVertexCountsAttr().Set([3, 3], 0)
    source.GetFaceVertexCountsAttr().Set([6], 0.5)
    with pytest.raises(RuntimeError, match="faceVertexCounts"):
        f2.UsdImporter().import_stage(stage, mesh_motion_options())
    source.GetFaceVertexCountsAttr().ClearAtTime(0.5)
    source.GetPointsAttr().Clear()
    source.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)] * 2, 0)
    source.GetNormalsAttr().Clear()
    assert not f2.UsdImporter().import_stage(stage, mesh_motion_options()).meshes[0].motion_times


@pytest.mark.parametrize("name", ["normals", "st", "temperature"])
def test_usd_requires_fixed_primvar_indices(name: str) -> None:
    stage = seam_stage()
    source = UsdGeom.Mesh(stage.GetPrimAtPath("/Seam"))
    positions = source.GetPointsAttr().Get(0)
    source.GetPointsAttr().Clear()
    source.GetPointsAttr().Set(positions)
    primvars = UsdGeom.PrimvarsAPI(source)
    if name == "normals":
        primvar = primvars.CreatePrimvar(name, Sdf.ValueTypeNames.Normal3fArray, "vertex")
        primvar.Set([(0, 0, 1)] * 3 + [(0, 1, 0)] * 3)
    elif name == "temperature":
        primvar = primvars.CreatePrimvar(name, Sdf.ValueTypeNames.FloatArray, "vertex")
        primvar.Set([0, 1, 2, 3, 4, 5])
    else:
        primvar = primvars.GetPrimvar(name)
    primvar.SetIndices([0, 1, 2, 3, 4, 5], 0)
    primvar.SetIndices([0, 1, 2, 3, 4, 5], 1)
    # Repeated identical indices are valid, even without position/normal time samples.
    f2.UsdImporter().import_stage(stage, mesh_motion_options())
    primvar.SetIndices([3, 1, 2, 0, 4, 5], 0.5)
    with pytest.raises(RuntimeError, match="changes indices"):
        f2.UsdImporter().import_stage(stage, mesh_motion_options())
    primvar.SetIndices([0, 1, 2, 3, 4, 5])
    f2.UsdImporter().import_stage(stage, f2.ImportOptions())


@pytest.mark.parametrize("primvar", [False, True])
def test_usd_animated_normals_with_static_positions(primvar: bool) -> None:
    stage = seam_stage()
    source = UsdGeom.Mesh(stage.GetPrimAtPath("/Seam"))
    positions = source.GetPointsAttr().Get(0)
    source.GetPointsAttr().Clear()
    source.GetPointsAttr().Set(positions)
    normals = (
        UsdGeom.PrimvarsAPI(source).CreatePrimvar(
            "normals", Sdf.ValueTypeNames.Normal3fArray, "vertex"
        )
        if primvar
        else source.GetNormalsAttr()
    )
    normals.Set([(0, 0, 1)] * 6, 0)
    normals.Set([(0, 0.6, 0.8)] * 6, 1)
    mesh = f2.UsdImporter().import_stage(stage, mesh_motion_options()).meshes[0]
    assert mesh.motion_times == [0, 1]
    indices = mesh.subgeometries[0].indices_numpy.ravel()
    np.testing.assert_allclose(_frames(mesh, P)[:, indices], [positions, positions])
    np.testing.assert_allclose(
        _frames(mesh, N)[:, indices], [normals.Get(0), normals.Get(1)], atol=1e-6
    )


def test_usd_uses_reference_primvars() -> None:
    stage = seam_stage()
    primvars = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/Seam"))
    ignored = primvars.CreatePrimvar("label", Sdf.ValueTypeNames.StringArray, "constant")
    ignored.Set(["a"], 0)
    ignored.Set(["b"], 1)
    imported = primvars.CreatePrimvar("temperature", Sdf.ValueTypeNames.FloatArray, "constant")
    imported.Set([0], 0)
    imported.Set([1], 1)
    mesh = f2.UsdImporter().import_stage(stage, mesh_motion_options()).meshes[0]
    assert len(mesh.motion_times) == 2
    np.testing.assert_array_equal(mesh.find_stream("temperature"), 0)


@pytest.mark.parametrize("attribute", ["points", "normals"])
def test_usd_rejects_changing_sample_counts(attribute: str) -> None:
    stage = seam_stage()
    source = stage.GetPrimAtPath("/Seam").GetAttribute(attribute)
    source.Set([(0, 0, 1)] * 6, 0)
    source.Set([(0, 0, 1)] * 3, 1)
    with pytest.raises(RuntimeError, match="changes .* count"):
        f2.UsdImporter().import_stage(stage, mesh_motion_options())


@pytest.mark.parametrize("scheme", ["none", "catmullClark", "bilinear", "loop"])
@pytest.mark.parametrize("refinement", [0, 2])
def test_usd_samples_match_static_meshes(scheme: str, refinement: int) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    source = UsdGeom.Mesh.Define(stage, "/Mesh")
    source.CreateSubdivisionSchemeAttr(scheme)
    faces = [[0, 1, 4, 3], [1, 2, 5, 4]]
    uvs = [[(0, 0), (1, 0), (1, 1), (0, 1)], [(2, 0), (3, 0), (3, 1), (2, 1)]]
    if scheme == "loop":
        faces = [[face[i] for i in tri] for face in faces for tri in ((0, 1, 2), (0, 2, 3))]
        uvs = [[uv[i] for i in tri] for uv in uvs for tri in ((0, 1, 2), (0, 2, 3))]
    source.CreateFaceVertexCountsAttr([len(face) for face in faces])
    source.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    st = UsdGeom.PrimvarsAPI(source).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
    )
    st.Set([uv for face in uvs for uv in face])
    UsdGeom.PrimvarsAPI(source).CreatePrimvar(
        "temperature", Sdf.ValueTypeNames.FloatArray, "vertex"
    ).Set([10, 20, 30, 40, 50, 60])
    points = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)])
    source.CreatePointsAttr().Set(points.tolist(), 0)
    points = points.astype(np.float32)
    points[4, 2] = 0.7
    points[2, 2] = -0.3
    source.GetPointsAttr().Set(points.tolist(), 1)
    options = mesh_motion_options()
    options.mesh_refinement_level = refinement
    mesh = f2.UsdImporter().import_stage(stage, options).meshes[0]
    assert mesh.properties["refinement_level"] == refinement
    indices = mesh.subgeometries[0].indices_numpy.ravel()
    if refinement and scheme != "none":
        assert len(indices) > sum((len(face) - 2) * 3 for face in faces)
    for sample, time in enumerate((0, 1)):
        snapshot = Usd.Stage.Open(stage.Flatten())
        snapshot_points = UsdGeom.Mesh(snapshot.GetPrimAtPath("/Mesh")).GetPointsAttr()
        snapshot_points.Clear()
        snapshot_points.Set(source.GetPointsAttr().Get(time))
        reference = f2.UsdImporter().import_stage(snapshot, options).meshes[0]
        reference_indices = reference.subgeometries[0].indices_numpy.ravel()
        for actual, expected in (
            (mesh.texcoords(), reference.texcoords()),
            (mesh.find_stream("temperature"), reference.find_stream("temperature")),
        ):
            np.testing.assert_allclose(actual[indices], expected[reference_indices], atol=1e-6)
        for semantic, expected in ((P, reference.positions), (N, reference.normals)):
            actual = mesh.find_stream(semantic, sample)
            np.testing.assert_allclose(actual[indices], expected[reference_indices], atol=1e-6)


def test_arm_against_usd() -> None:
    path = ASSETS / "arm/arm_baked.usda"
    mesh = f2.UsdImporter().load_scene(path, mesh_motion_options()).meshes[0]
    stage = Usd.Stage.Open(str(path))
    source = UsdGeom.Mesh(stage.GetPrimAtPath(mesh.name))
    times = source.GetPointsAttr().GetTimeSamples()
    expected = np.array([source.GetPointsAttr().Get(time) for time in times])
    p = _frames(mesh, P)
    mapping = np.argmin(np.linalg.norm(p[0, :, None] - expected[0, None], axis=-1), axis=-1)
    np.testing.assert_allclose(p, expected[:, mapping], atol=1e-6)
    np.testing.assert_allclose(mesh.motion_times, np.array(times) / stage.GetTimeCodesPerSecond())
    assert not np.array_equal(p[0], p[-1])


@pytest.mark.parametrize("normal_mode", ["vertex", "indexed_face_varying", "generated"])
def test_skinning_preserves_mesh_sample_times(normal_mode: str) -> None:
    from falcor2.pxr import Gf, UsdSkel

    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(1)
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(20)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdSkel.Root.Define(stage, "/Character")
    root.AddTranslateOp().Set((10, 0, 0))
    skeleton = UsdSkel.Skeleton.Define(stage, "/Character/Skeleton")
    skeleton.AddTranslateOp().Set((0, 5, 0))
    skeleton.CreateJointsAttr(["a", "b"])
    skeleton.CreateBindTransformsAttr([Gf.Matrix4d(1)] * 2)
    skeleton.CreateRestTransformsAttr([Gf.Matrix4d(1)] * 2)
    animation = UsdSkel.Animation.Define(stage, "/Character/Animation")
    animation.CreateJointsAttr(["a", "b"])
    animation.CreateScalesAttr([(1, 1, 1)] * 2)
    for time in range(1, 21):
        animation.CreateTranslationsAttr().Set([(time, 0, 0), (0, 0, time / 2)], time)
        rotation = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), time * 3).GetQuat())
        animation.CreateRotationsAttr().Set([rotation, Gf.Quatf(1)], time)
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets(
        [animation.GetPath()]
    )
    source = UsdGeom.Mesh.Define(stage, "/Character/Body")
    source.AddTranslateOp().Set((2, 0, 0))
    source.CreateSubdivisionSchemeAttr("catmullClark" if normal_mode == "generated" else "none")
    source.CreateFaceVertexCountsAttr([3, 3])
    source.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    points = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], dtype=np.float32)
    for time in (1, 10, 20):
        source.CreatePointsAttr().Set((points * (1 + time / 20)).tolist(), time)
    if normal_mode == "vertex":
        source.CreateNormalsAttr([(0, 0, 1)] * 4)
        source.SetNormalsInterpolation("vertex")
    elif normal_mode == "indexed_face_varying":
        normals = UsdGeom.PrimvarsAPI(source).CreatePrimvar(
            "normals", Sdf.ValueTypeNames.Normal3fArray, "faceVarying"
        )
        normals.SetIndices([0, 1, 0, 0, 0, 1])
        normals.Set([(0, 0, 1), (0, 0.6, 0.8)], 1)
        normals.Set([(0, 0.6, 0.8), (0, 0, 1)], 20)
    binding = UsdSkel.BindingAPI.Apply(source.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateJointsAttr(["b", "a"])
    binding.CreateGeomBindTransformAttr(Gf.Matrix4d().SetTranslate((2, 0, 0)))
    binding.CreateJointIndicesPrimvar(False, 1).Set([0, 1, 0, 1])
    binding.CreateJointWeightsPrimvar(False, 1).Set([1] * 4)
    unrelated = UsdGeom.Xform.Define(stage, "/Moving")
    translation = unrelated.AddTranslateOp()
    translation.Set((0, 0, 0), 1)
    translation.Set((0, 0, 3), 20)

    original = stage.GetRootLayer().ExportToString()
    session = stage.GetSessionLayer().ExportToString()
    options = mesh_motion_options()
    options.bake_poses = True
    imported = f2.UsdImporter().import_stage(stage, options)
    mesh = imported.meshes[0]
    assert mesh.motion_times == [1, 10, 20]
    assert len(imported.animation.channels) > 0
    assert mesh.properties["refinement_level"] == 0
    indices = mesh.subgeometries[0].indices_numpy.ravel()
    static_options = f2.ImportOptions()
    static_options.bake_poses = True
    for sample, time in enumerate((1, 10, 20)):
        snapshot = Usd.Stage.Open(stage.Flatten())
        for prim in snapshot.Traverse():
            for attribute in prim.GetAttributes():
                if attribute.GetNumTimeSamples():
                    value = attribute.Get(time)
                    attribute.Clear()
                    attribute.Set(value)
        reference = f2.UsdImporter().import_stage(snapshot, static_options).meshes[0]
        reference_indices = reference.subgeometries[0].indices_numpy.ravel()
        for semantic, expected in ((P, reference.positions), (N, reference.normals)):
            np.testing.assert_allclose(
                mesh.find_stream(semantic, sample)[indices], expected[reference_indices], atol=2e-6
            )
    assert stage.GetRootLayer().ExportToString() == original
    assert stage.GetSessionLayer().ExportToString() == session

    # Skeleton animation alone does not create mesh samples; evaluate it at numeric zero.
    source.GetPointsAttr().Clear()
    source.GetPointsAttr().Set(points.tolist())
    if normal_mode == "indexed_face_varying":
        normals.GetAttr().Clear()
        normals.Set([(0, 0, 1), (0, 0.6, 0.8)])
    mesh = f2.UsdImporter().import_stage(stage, options).meshes[0]
    reference = f2.UsdImporter().import_stage(stage, static_options).meshes[0]
    assert not mesh.motion_times
    np.testing.assert_allclose(mesh.positions, reference.positions, atol=2e-6)


@pytest.mark.parametrize("normal_scale", [1, 2])
def test_usd_indexed_normals_follow_tangent_splits(normal_scale: float) -> None:
    stage = Usd.Stage.CreateInMemory()
    source = UsdGeom.Mesh.Define(stage, "/Quad")
    source.CreateSubdivisionSchemeAttr("none")
    source.CreateFaceVertexCountsAttr([4])
    source.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    points = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], dtype=np.float32)
    source.CreatePointsAttr().Set(points.tolist(), 0)
    moved = points[:, [1, 0, 2]].copy()
    moved[:, 0] *= -1
    source.GetPointsAttr().Set(moved.tolist(), 1)
    primvars = UsdGeom.PrimvarsAPI(source)
    uv = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    uv.Set([(0, 0), (1, 0), (1, 1), (1, 0)])
    normals = primvars.CreatePrimvar("normals", Sdf.ValueTypeNames.Normal3fArray, "faceVarying")
    normals.SetIndices([0, 1, 0, 1])
    normals.Set([(0, 0, normal_scale), (0, 0, normal_scale)], 0)
    normals.Set([(0, 0, normal_scale), (0, 0.6 * normal_scale, 0.8 * normal_scale)], 1)
    mesh = f2.UsdImporter().import_stage(stage, mesh_motion_options()).meshes[0]
    mapping = [0, 1, 2, 3, 0, 2]
    np.testing.assert_allclose(_frames(mesh, P), np.array([points[mapping], moved[mapping]]))
    expected_normals = np.asarray(normals.ComputeFlattened(1))[mapping]
    np.testing.assert_allclose(_frames(mesh, N)[1], expected_normals, atol=1e-6)
    for sample in (0, 1):
        snapshot = Usd.Stage.Open(stage.Flatten())
        snapshot_mesh = UsdGeom.Mesh(snapshot.GetPrimAtPath("/Quad"))
        for attribute in (source.GetPointsAttr(), normals.GetAttr()):
            target = snapshot_mesh.GetPrim().GetAttribute(attribute.GetName())
            target.Clear()
            target.Set(attribute.Get(sample))
        reference = f2.UsdImporter().import_stage(snapshot).meshes[0]
        np.testing.assert_allclose(
            _frames(mesh, N)[sample][mesh.subgeometries[0].indices_numpy.ravel()],
            reference.normals[reference.subgeometries[0].indices_numpy.ravel()],
            atol=1e-6,
        )
    expected_normals = expected_normals / np.linalg.norm(expected_normals, axis=-1, keepdims=True)
    expected_tangents = _frames(mesh, T)[0][:, [1, 0, 2]].copy()
    expected_tangents[:, 0] *= -1
    expected_tangents -= expected_normals * np.sum(
        expected_tangents * expected_normals, axis=-1, keepdims=True
    )
    expected_tangents /= np.linalg.norm(expected_tangents, axis=-1, keepdims=True)
    np.testing.assert_allclose(_frames(mesh, T)[1], expected_tangents, atol=1e-6)
