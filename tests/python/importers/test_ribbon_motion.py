# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import falcor2 as f2
from falcor2.pxr import Usd, UsdGeom, Sdf

P, N, T = (f2.ImporterSemantic.position, f2.ImporterSemantic.normal, f2.ImporterSemantic.tangent)


def _options(refinement: int = 2) -> f2.ImportOptions:
    options = f2.ImportOptions()
    options.import_curve_motion = True
    options.curve_refinement_level = refinement
    return options


def _stage(curve_type: str = "linear", basis: str = "bezier", wrap: str = "nonperiodic") -> tuple:
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(2)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    curve = UsdGeom.BasisCurves.Define(stage, "/Ribbon")
    curve.CreateTypeAttr(curve_type)
    curve.CreateBasisAttr(basis)
    curve.CreateWrapAttr(wrap)
    count = 6 if curve_type == "cubic" and basis == "bezier" and wrap == "periodic" else 4
    points = np.zeros((count, 3), dtype=np.float32)
    if wrap == "periodic":
        a = np.arange(count) * 2 * np.pi / count
        points[:, :2] = np.column_stack((np.cos(a), np.sin(a))) * 3
    else:
        points[:, 0] = np.arange(count)
    curve.CreateCurveVertexCountsAttr([count])
    curve.CreatePointsAttr(points.tolist())
    curve.GetPointsAttr().Set(points.tolist(), 0)
    moved = points.copy()
    moved[:, 2] = 0.4
    curve.GetPointsAttr().Set(moved.tolist(), 2)
    curve.CreateNormalsAttr([(0, 0, 1)])
    curve.SetNormalsInterpolation("constant")
    curve.GetNormalsAttr().Set([(0, 0, 1)], 0.5)
    curve.GetNormalsAttr().Set([(0, 0.3, 1)], 1.5)
    curve.CreateWidthsAttr([0.2])
    curve.SetWidthsInterpolation("constant")
    curve.GetWidthsAttr().Set([0.2], 0.25)
    curve.GetWidthsAttr().Set([0.4], 1.75)
    return stage, curve


def _freeze(stage: Usd.Stage, time: float) -> Usd.Stage:
    result = Usd.Stage.Open(stage.Flatten())
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetNumTimeSamples():
                target = result.GetPrimAtPath(prim.GetPath()).GetAttribute(attr.GetName())
                target.Clear()
                target.Set(attr.Get(time))
    return result


@pytest.mark.parametrize(
    "curve_type,basis,wrap",
    [
        ("linear", "bezier", "nonperiodic"),
        ("linear", "bezier", "periodic"),
        ("cubic", "bezier", "nonperiodic"),
        ("cubic", "bezier", "periodic"),
        ("cubic", "bspline", "nonperiodic"),
        ("cubic", "bspline", "pinned"),
        ("cubic", "bspline", "periodic"),
        ("cubic", "catmullRom", "nonperiodic"),
        ("cubic", "catmullRom", "pinned"),
        ("cubic", "catmullRom", "periodic"),
    ],
)
@pytest.mark.parametrize("refinement", [0, 3])
def test_ribbon_motion_matches_frozen_samples(
    curve_type: str, basis: str, wrap: str, refinement: int
) -> None:
    stage, _ = _stage(curve_type, basis, wrap)
    options = _options(refinement)
    imported = f2.UsdImporter().import_stage(stage, options)
    assert not imported.curves
    mesh = imported.meshes[0]
    times = [0, 0.25, 0.5, 1.5, 1.75, 2]
    np.testing.assert_allclose(mesh.motion_times, np.array(times) / 2)
    indices = mesh.subgeometries[0].indices_numpy
    for sample, time in enumerate(times):
        static_options = _options(refinement)
        static_options.import_curve_motion = False
        reference = f2.UsdImporter().import_stage(_freeze(stage, time), static_options).meshes[0]
        reference_indices = reference.subgeometries[0].indices_numpy
        for semantic in (P, N):
            np.testing.assert_allclose(
                mesh.find_stream(semantic, sample)[indices],
                reference.find_stream(semantic)[reference_indices],
                atol=2e-6,
            )
        np.testing.assert_allclose(
            mesh.texcoords(0)[indices], reference.texcoords(0)[reference_indices], atol=1e-6
        )
        n, t = mesh.find_stream(N, sample), mesh.find_stream(T, sample)
        assert np.isfinite(t).all()
        np.testing.assert_allclose(np.linalg.norm(t, axis=-1), 1, atol=2e-6)
        np.testing.assert_allclose(np.sum(n * t, axis=-1), 0, atol=2e-6)


@pytest.mark.parametrize("primvars", [False, True])
def test_ribbon_normal_only_motion_and_option_routing(primvars: bool) -> None:
    stage, curve = _stage()
    for attr in (curve.GetPointsAttr(), curve.GetWidthsAttr()):
        value = attr.Get(0)
        attr.Clear()
        attr.Set(value)
    if primvars:
        normals = UsdGeom.PrimvarsAPI(curve).CreatePrimvar(
            "normals", Sdf.ValueTypeNames.Normal3fArray, "constant"
        )
        normals.SetIndices([0])
        normals.Set([(0, 0, 1)], 3)
        normals.Set([(0, 1, 1)], 5)
        normals.Set([(0, 0, 1)])
        expected_times = [1.5, 2.5]
    else:
        expected_times = [0.25, 0.75]
    stage.SetInterpolationType(Usd.InterpolationTypeHeld)
    options = _options()
    mesh = f2.UsdImporter().import_stage(stage, options).meshes[0]
    assert mesh.motion_times == expected_times
    assert mesh.motion_interpolation_mode == f2.AnimationInterpolationMode.step
    assert not np.allclose(mesh.find_stream(P, 0), mesh.find_stream(P, 1))
    for enable_mesh in (False, True):
        options.import_curve_motion = False
        options.import_mesh_motion = enable_mesh
        ordinary = f2.UsdImporter().import_stage(stage, options).meshes[0]
        assert not ordinary.motion_times
        np.testing.assert_allclose(ordinary.normals, np.tile([0, 0, 1], (ordinary.vertex_count, 1)))
    for rate in (0, -1):
        stage.SetTimeCodesPerSecond(rate)
        mesh = f2.UsdImporter().import_stage(stage, _options()).meshes[0]
        np.testing.assert_allclose(mesh.motion_times, np.array(expected_times) * 2 / 24)


@pytest.mark.parametrize("recompute", [False, True])
def test_ribbon_motion_preserves_collapsed_widths(recompute: bool) -> None:
    stage, curve = _stage()
    curve.GetCurveVertexCountsAttr().Set([2])
    for attr, value in (
        (curve.GetPointsAttr(), [(0, 0, 0), (2, 0, 0)]),
        (curve.GetNormalsAttr(), [(0, 0, 1)]),
    ):
        attr.Clear()
        attr.Set(value)
    curve.GetWidthsAttr().Clear()
    curve.GetWidthsAttr().Set([0], 0)
    curve.GetWidthsAttr().Set([1], 1)
    options = _options(0)
    options.recompute_normals = recompute
    mesh = f2.UsdImporter().import_stage(stage, options).meshes[0]
    assert mesh.vertex_count == 4
    assert len(mesh.subgeometries[0].indices_numpy) == 2
    np.testing.assert_allclose(mesh.find_stream(P, 0)[:, 1], 0)
    assert np.max(np.abs(mesh.find_stream(P, 1)[:, 1])) == pytest.approx(0.5)
    for sample in (0, 1):
        for semantic in (P, N, T):
            assert np.isfinite(mesh.find_stream(semantic, sample)).all()


def test_ribbon_motion_deduplication_keeps_separating_strands() -> None:
    stage, curve = _stage()
    curve.GetCurveVertexCountsAttr().Set([2, 2])
    points = np.array([(0, 0, 0), (2, 0, 0)] * 2, dtype=np.float32)
    curve.GetPointsAttr().Clear()
    curve.GetPointsAttr().Set(points.tolist(), 0)
    points[2:, 1] += 3
    curve.GetPointsAttr().Set(points.tolist(), 1)
    for attr, value in ((curve.GetNormalsAttr(), [(0, 0, 1)]), (curve.GetWidthsAttr(), [0.2])):
        attr.Clear()
        attr.Set(value)
    mesh = f2.UsdImporter().import_stage(stage, _options(0)).meshes[0]
    assert mesh.vertex_count == 8
    assert len(mesh.subgeometries[0].indices_numpy) == 4
    assert len(np.unique(mesh.find_stream(P, 0), axis=0)) == 4
    assert len(np.unique(mesh.find_stream(P, 1), axis=0)) == 8


def test_ribbon_motion_indexed_primvars_and_static_uvs() -> None:
    stage, curve = _stage()
    for attr in (curve.GetPointsAttr(), curve.GetNormalsAttr()):
        value = attr.Get(0)
        attr.Clear()
        attr.Set(value)
    api = UsdGeom.PrimvarsAPI(curve)
    widths = api.CreatePrimvar("widths", Sdf.ValueTypeNames.FloatArray, "vertex")
    widths.SetIndices([1, 0, 1, 0])
    widths.Set([0.2, 0.4], 3)
    widths.Set([0.4, 0.8], 5)
    st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    st.Set([(0.2, 0.3), (0.8, 0.7)])
    st.SetIndices([1, 0, 1, 0])
    mesh = f2.UsdImporter().import_stage(stage, _options(0)).meshes[0]
    assert mesh.motion_times == [1.5, 2.5]
    np.testing.assert_allclose(
        np.abs(mesh.find_stream(P, 1)[:, 1]), 2 * np.abs(mesh.positions[:, 1])
    )
    for sample, time in enumerate((3, 5)):
        reference = f2.UsdImporter().import_stage(_freeze(stage, time)).meshes[0]
        np.testing.assert_allclose(mesh.texcoords(0), reference.texcoords(0))
        np.testing.assert_allclose(mesh.find_stream(P, sample), reference.positions)


@pytest.mark.parametrize(
    "name",
    [
        "points",
        "normals",
        "widths",
        "curveVertexCounts",
        "normals:indices",
        "widths:indices",
        "st:indices",
    ],
)
def test_ribbon_motion_rejects_changing_layout(name: str) -> None:
    stage, curve = _stage()
    if name.endswith(":indices"):
        base = name.split(":")[0]
        type_name, values = {
            "normals": (Sdf.ValueTypeNames.Normal3fArray, [(0, 0, 1), (0, 1, 1)]),
            "widths": (Sdf.ValueTypeNames.FloatArray, [0.2, 0.4]),
            "st": (Sdf.ValueTypeNames.TexCoord2fArray, [(0, 0), (1, 1)]),
        }[base]
        primvar = UsdGeom.PrimvarsAPI(curve).CreatePrimvar(base, type_name, "vertex")
        primvar.Set(values)
        primvar.SetIndices([0, 1, 0, 1], 0)
        primvar.SetIndices([1, 0, 1, 0], 1)
    else:
        attr = curve.GetPrim().GetAttribute(name)
        value = attr.Get(0)
        attr.Set(value, 0)
        attr.Set([2, 2] if name == "curveVertexCounts" else list(value) + [value[0]], 1)
    with pytest.raises(RuntimeError, match="changes|count"):
        f2.UsdImporter().import_stage(stage, _options())


@pytest.mark.parametrize("recompute", [False, True])
def test_ribbon_motion_time_samples_without_defaults(recompute: bool) -> None:
    stage, curve = _stage()
    for attr in (curve.GetPointsAttr(), curve.GetNormalsAttr(), curve.GetWidthsAttr()):
        attr.ClearAtTime(Usd.TimeCode.Default())
    options = _options()
    options.recompute_normals = recompute
    mesh = f2.UsdImporter().import_stage(stage, options).meshes[0]
    assert len(mesh.motion_times) == 6
    for sample in range(6):
        for semantic in (P, N, T):
            assert np.isfinite(mesh.find_stream(semantic, sample)).all()


def test_ribbon_motion_retains_coincident_edges_that_open() -> None:
    stage, curve = _stage()
    curve.GetPointsAttr().Clear()
    curve.GetPointsAttr().Set([(0, 0, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0)], 0)
    curve.GetPointsAttr().Set([(0, 0, 0), (0.5, 0, 0), (1, 0, 0), (2, 0, 0)], 2)
    for attr, value in ((curve.GetNormalsAttr(), [(0, 0, 1)]), (curve.GetWidthsAttr(), [0.2])):
        attr.Clear()
        attr.Set(value)
    mesh = f2.UsdImporter().import_stage(stage, _options(0)).meshes[0]
    assert len(mesh.subgeometries[0].indices_numpy) == 6
    assert mesh.vertex_count == 8
    reference = f2.UsdImporter().import_stage(_freeze(stage, 2)).meshes[0]
    indices = mesh.subgeometries[0].indices_numpy
    np.testing.assert_allclose(
        mesh.find_stream(P, 1)[indices],
        reference.positions[reference.subgeometries[0].indices_numpy],
    )


def test_ribbon_motion_repairs_each_samples_spatial_frames() -> None:
    stage, curve = _stage()
    for attr in (curve.GetPointsAttr(), curve.GetWidthsAttr()):
        value = attr.Get(0)
        attr.Clear()
        attr.Set(value)
    curve.GetNormalsAttr().Clear()
    curve.SetNormalsInterpolation("vertex")
    curve.GetNormalsAttr().Set([(0, 0, 1), (0, 0, 0), (0, 0, 1), (0, 0, 1)], 0)
    curve.GetNormalsAttr().Set([(0, 1, 1), (0, 0, 1), (0, 0, 0), (0, 0, 1)], 2)
    mesh = f2.UsdImporter().import_stage(stage, _options(0)).meshes[0]
    for sample, time in enumerate((0, 2)):
        reference = f2.UsdImporter().import_stage(_freeze(stage, time)).meshes[0]
        indices = mesh.subgeometries[0].indices_numpy
        for semantic in (P, N):
            np.testing.assert_allclose(
                mesh.find_stream(semantic, sample)[indices],
                reference.find_stream(semantic)[reference.subgeometries[0].indices_numpy],
                atol=2e-6,
            )


def test_ribbon_motion_rejects_normal_indices_that_change_geometry_kind() -> None:
    stage, curve = _stage()
    normals = UsdGeom.PrimvarsAPI(curve).CreatePrimvar(
        "normals", Sdf.ValueTypeNames.Normal3fArray, "constant"
    )
    normals.Set([(0, 0, 1)])
    normals.SetIndices([], 0)
    normals.SetIndices([0], 2)
    with pytest.raises(RuntimeError, match="changes indices"):
        f2.UsdImporter().import_stage(stage, _options())


@pytest.mark.parametrize("curve_type", ["linear", "cubic"])
@pytest.mark.parametrize("refinement", [0, 3])
@pytest.mark.parametrize("interpolation", ["vertex", "varying"])
@pytest.mark.parametrize("motion", [False, True])
def test_pinned_linear_and_bezier_match_nonperiodic(
    curve_type: str, refinement: int, interpolation: str, motion: bool
) -> None:
    stage = Usd.Stage.CreateInMemory()
    curve = UsdGeom.BasisCurves.Define(stage, "/Ribbon")
    curve.CreateTypeAttr(curve_type)
    curve.CreateBasisAttr("bezier")
    curve.CreateWrapAttr("nonperiodic")
    count = 3 if curve_type == "linear" else 7
    curve.CreateCurveVertexCountsAttr([count])
    points = [(float(i), 0, 0) for i in range(count)]
    curve.CreatePointsAttr(points)
    attribute_count = count if interpolation == "vertex" else 3
    widths = [0.2 + 0.05 * i for i in range(attribute_count)]
    curve.CreateWidthsAttr(widths)
    curve.SetWidthsInterpolation(interpolation)
    curve.CreateNormalsAttr([(0, 0.1 * i, 1) for i in range(attribute_count)])
    curve.SetNormalsInterpolation(interpolation)
    st = UsdGeom.PrimvarsAPI(curve).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation
    )
    st.Set([(0.1 * i, 0.2 * i) for i in range(attribute_count)])
    if motion:
        curve.GetPointsAttr().Set(points, 0)
        curve.GetPointsAttr().Set([(x, y, 1) for x, y, z in points], 1)
        curve.GetWidthsAttr().Set(widths, 0)
        curve.GetWidthsAttr().Set([2 * w for w in widths], 1)
    options = _options(refinement)
    options.import_curve_motion = motion
    reference = f2.UsdImporter().import_stage(stage, options).meshes[0]
    curve.GetWrapAttr().Set("pinned")
    mesh = f2.UsdImporter().import_stage(stage, options).meshes[0]
    assert mesh.motion_times == reference.motion_times
    np.testing.assert_array_equal(
        mesh.subgeometries[0].indices_numpy, reference.subgeometries[0].indices_numpy
    )
    np.testing.assert_allclose(mesh.texcoords(0), reference.texcoords(0))
    for sample in range(2 if motion else 1):
        for semantic in (P, N, T):
            np.testing.assert_allclose(
                mesh.find_stream(semantic, sample), reference.find_stream(semantic, sample)
            )


@pytest.mark.parametrize("count", [3, 5])
def test_pinned_bezier_rejects_invalid_control_counts(count: int) -> None:
    stage, curve = _stage("cubic", "bezier", "pinned")
    curve.GetCurveVertexCountsAttr().Set([count])
    curve.GetPointsAttr().Clear()
    curve.GetPointsAttr().Set([(float(i), 0, 0) for i in range(count)])
    with pytest.raises(RuntimeError, match="invalid Bezier topology"):
        f2.UsdImporter().import_stage(stage, _options())
