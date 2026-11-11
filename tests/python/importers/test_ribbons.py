# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import falcor2 as f2
from falcor2.pxr import Usd, UsdGeom, Sdf


@pytest.mark.parametrize("name,y", [("LinearTwist", 0), ("CubicTwist", -3)])
def test_usd_ribbon_full_turn(name: str, y: float) -> None:
    path = Path(__file__).resolve().parents[3] / "data/assets/test_usd_ribbons/twist.usda"
    stage = Usd.Stage.Open(str(path))
    stage.SetEditTarget(stage.GetSessionLayer())
    for prim in stage.GetPseudoRoot().GetChildren():
        prim.SetActive(prim.GetName() == name)
    options = f2.ImportOptions()
    options.curve_refinement_level = 3
    imported = f2.UsdImporter().import_stage(stage, options)
    assert len(imported.meshes) == 1
    mesh = imported.meshes[0]

    # Four intervals per span include interior normals in every quarter turn.
    # Compare actual edge vertices, independent of mesh deduplication order.
    xs = np.repeat(np.linspace(0, 8, 17), 2)
    authored = np.array([[0, 0, 1], [0, -1, 0], [0, 0, -1], [0, 1, 0], [0, 0, 1]])
    span = np.minimum((xs / 2).astype(int), 3)
    t = (xs / 2 - span)[:, None]
    normals = authored[span] * (1 - t) + authored[span + 1] * t
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    sides = np.cross(normals, [1, 0, 0])
    centers = np.column_stack([xs, np.full_like(xs, y), np.zeros_like(xs)])
    positions = centers + np.tile([-0.6, 0.6], 17)[:, None] * sides
    uv = mesh.texcoords(0)
    order = np.lexsort((uv[:, 0], uv[:, 1]))
    assert mesh.vertex_count == 34
    assert len(mesh.subgeometries[0].indices_numpy) == 32
    np.testing.assert_allclose(mesh.positions[order], positions, atol=2e-6)
    np.testing.assert_allclose(mesh.normals[order], normals, atol=2e-6)
    np.testing.assert_allclose(uv[order], np.column_stack([np.tile([0, 1], 17), xs / 8]))


@pytest.mark.parametrize("case", ["twist", "joint", "repair"])
@pytest.mark.parametrize("left_handed", [False, True])
def test_usd_ribbon_sampled_normal_field(case: str, left_handed: bool) -> None:
    points = np.array([[0, 0, 0], [3, 0, 0], [3, 1, 0]], dtype=float)
    tangents = np.array([[1, 0, 0], [2**-0.5, 2**-0.5, 0], [0, 1, 0]])
    authored = np.array([[0.2, 0, 1], [0, 0, 1], [0, 0.4, 1]])
    if case != "joint":
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        tangents = np.tile([1, 0, 0], (3, 1))
        authored = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=float)
    expected_normals = authored / np.linalg.norm(authored, axis=1, keepdims=True)
    if case == "repair":
        authored[1] = 0
    sides = np.cross(expected_normals, tangents)
    sides /= np.linalg.norm(sides, axis=1, keepdims=True)
    vertices = np.stack([points - 0.1 * sides, points + 0.1 * sides], axis=1).reshape(-1, 3)
    vertex_normals = np.repeat(expected_normals, 2, axis=0)
    triangles = np.array([[0, 2, 1], [1, 2, 3], [2, 4, 3], [3, 4, 5]])
    if left_handed:
        triangles = triangles[:, [0, 2, 1]]

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    curve = UsdGeom.BasisCurves.Define(stage, "/Strip")
    curve.CreateTypeAttr("linear")
    curve.CreateOrientationAttr("leftHanded" if left_handed else "rightHanded")
    curve.CreateCurveVertexCountsAttr([3])
    curve.CreatePointsAttr(points.tolist())
    curve.CreateNormalsAttr(authored.tolist())
    curve.SetNormalsInterpolation("vertex")
    curve.CreateWidthsAttr([0.2])
    curve.SetWidthsInterpolation("constant")
    options = f2.ImportOptions()
    options.curve_refinement_level = 0
    mesh = f2.UsdImporter().import_stage(stage, options).meshes[0]
    indices = mesh.subgeometries[0].indices_numpy
    # The imported vertex field determines ordinary mesh shading interpolation;
    # no ribbon-specific shader or GPU scene is needed to validate it.
    assert mesh.vertex_count == 6
    np.testing.assert_allclose(mesh.positions[indices], vertices[triangles], atol=2e-6)
    np.testing.assert_allclose(mesh.normals[indices], vertex_normals[triangles], atol=2e-6)
    assert np.isfinite(mesh.tangents).all()


@pytest.mark.parametrize(
    "curve_type,basis,wrap,refinement",
    [("linear", "bezier", "nonperiodic", 0)]
    + [
        ("cubic", basis, wrap, refinement)
        for basis in ("bezier", "bspline", "catmullRom")
        for wrap in ("nonperiodic", "pinned")
        for refinement in (0, 3)
    ],
)
@pytest.mark.parametrize("interpolation", ["vertex", "varying"])
def test_ribbon_samples_match_round_curve_evaluation(
    curve_type: str, basis: str, wrap: str, refinement: int, interpolation: str
) -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(4)
    curve = UsdGeom.BasisCurves.Define(stage, "/Curve")
    curve.CreateTypeAttr(curve_type)
    curve.CreateBasisAttr(basis)
    curve.CreateWrapAttr(wrap)
    curve.CreateCurveVertexCountsAttr([7])
    points = np.column_stack([np.arange(7), np.sin(np.arange(7)), np.zeros(7)])
    curve.CreatePointsAttr(points.tolist())
    curve.GetPointsAttr().Set(points.tolist(), 0)
    points[:, 1] *= 0.5
    curve.GetPointsAttr().Set(points.tolist(), 4)
    # Ignore an animated builtin width when an indexed primvar takes precedence.
    curve.CreateWidthsAttr().Set([99], 99)
    spans = (
        6
        if curve_type == "linear" or (wrap == "pinned" and basis != "bezier")
        else 2 if basis == "bezier" else 4
    )
    count = 7 if interpolation == "vertex" else spans + 1
    api = UsdGeom.PrimvarsAPI(curve)
    widths = api.CreatePrimvar("widths", Sdf.ValueTypeNames.FloatArray, interpolation)
    widths.SetIndices([i % 2 for i in range(count)])
    widths.Set([0.2, 0.4], 1)
    widths.Set([0.4, 0.6], 3)
    st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation)
    st.Set([(i / (count - 1), 0.25) for i in range(count)])
    options = f2.ImportOptions()
    options.curve_refinement_level = refinement
    options.import_curve_motion = True
    rounded = f2.UsdImporter().import_stage(stage, options).curves[0]
    curve.CreateNormalsAttr([(0, 0, 1)])
    curve.SetNormalsInterpolation("constant")
    ribbon = f2.UsdImporter().import_stage(stage, options).meshes[0]
    assert ribbon.motion_times == rounded.motion_times == [0, 0.25, 0.75, 1]
    assert ribbon.motion_interpolation_mode == rounded.motion_interpolation_mode
    # Authored U increases along the strand and is copied to both ribbon edges.
    order = np.argsort(ribbon.texcoords(0)[:, 0], kind="stable")
    uv_pairs = ribbon.texcoords(0)[order].reshape(-1, 2, 2)
    np.testing.assert_allclose(uv_pairs[:, 0], rounded.texcoords, atol=2e-6)
    np.testing.assert_allclose(uv_pairs[:, 1], rounded.texcoords, atol=2e-6)
    for sample in range(len(ribbon.motion_times)):
        pairs = ribbon.find_stream(f2.ImporterSemantic.position, sample)[order].reshape(-1, 2, 3)
        np.testing.assert_allclose(pairs.mean(axis=1), rounded.position_sample(sample), atol=2e-6)
        np.testing.assert_allclose(
            np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1),
            2 * rounded.radius_sample(sample),
            atol=2e-6,
        )


@pytest.mark.parametrize("primvar", [False, True])
@pytest.mark.parametrize("initially_oriented", [False, True])
def test_curve_motion_rejects_representation_changes(
    primvar: bool, initially_oriented: bool
) -> None:
    stage = Usd.Stage.CreateInMemory()
    curve = UsdGeom.BasisCurves.Define(stage, "/Curve")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr([(0, 0, 0), (1, 0, 0)])
    if primvar:
        normals = UsdGeom.PrimvarsAPI(curve).CreatePrimvar(
            "normals", Sdf.ValueTypeNames.Normal3fArray, "constant"
        )
    else:
        normals = curve.CreateNormalsAttr()
        curve.SetNormalsInterpolation("constant")
    normals.Set([(0, 0, 1)] if initially_oriented else [], 0)
    normals.Set([] if initially_oriented else [(0, 0, 1)], 1)
    options = f2.ImportOptions()
    options.import_curve_motion = True
    with pytest.raises(RuntimeError, match="changes value count"):
        f2.UsdImporter().import_stage(stage, options)


@pytest.mark.parametrize("oriented", [False, True])
def test_inherited_normals_choose_curve_output(oriented: bool) -> None:
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/Parent")
    normals = UsdGeom.PrimvarsAPI(parent).CreatePrimvar(
        "normals", Sdf.ValueTypeNames.Normal3fArray, "constant"
    )
    normals.Set([(0, 0, 1)] if oriented else [])
    curve = UsdGeom.BasisCurves.Define(stage, "/Parent/Curve")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr([(0, 0, 0), (1, 0, 0)])
    # The inherited primvar determines dispatch, including an explicitly empty value.
    curve.CreateNormalsAttr([(0, 1, 0)])
    curve.SetNormalsInterpolation("constant")
    scene = f2.UsdImporter().import_stage(stage)
    assert len(scene.meshes) == int(oriented)
    assert len(scene.curves) == int(not oriented)
    if oriented:
        np.testing.assert_allclose(
            scene.meshes[0].normals, np.tile([0, 0, 1], (scene.meshes[0].vertex_count, 1))
        )


@pytest.mark.parametrize("curve_type", ["linear", "cubic"])
@pytest.mark.parametrize("wrap", ["nonperiodic", "periodic"])
def test_round_sampling_compatibility_with_shared_evaluator(curve_type: str, wrap: str) -> None:
    stage = Usd.Stage.CreateInMemory()
    curve = UsdGeom.BasisCurves.Define(stage, "/Curve")
    curve.CreateTypeAttr(curve_type)
    curve.CreateBasisAttr("bspline")
    curve.CreateWrapAttr(wrap)
    curve.CreateCurveVertexCountsAttr([4])
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    curve.CreatePointsAttr(points.tolist())
    curve.CreateWidthsAttr([0.2])
    curve.SetWidthsInterpolation("constant")
    options = f2.ImportOptions()
    options.curve_refinement_level = 3
    rounded = f2.UsdImporter().import_stage(stage, options).curves[0]
    # Legacy round curves ignore linear refinement and never close periodic geometry.
    expected_count = 4 if curve_type == "linear" else 5
    assert len(rounded.positions) == expected_count
    assert len(rounded.indices) == 2 * (expected_count - 1)
    if curve_type == "linear":
        np.testing.assert_array_equal(rounded.positions, points)
    else:
        np.testing.assert_allclose(
            rounded.positions[0], (points[0] + 4 * points[1] + points[2]) / 6, atol=1e-6
        )
        np.testing.assert_allclose(
            rounded.positions[-1], (points[1] + 4 * points[2] + points[3]) / 6, atol=1e-6
        )
    curve.CreateNormalsAttr([(0, 0, 1)])
    curve.SetNormalsInterpolation("constant")
    ribbon = f2.UsdImporter().import_stage(stage, options).meshes[0]
    spans = 4 if wrap == "periodic" else 3 if curve_type == "linear" else 1
    assert ribbon.vertex_count == 2 * (4 * spans + 1)
    assert len(ribbon.subgeometries[0].indices_numpy) == 2 * 4 * spans
