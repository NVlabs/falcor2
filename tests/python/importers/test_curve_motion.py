# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import falcor2 as f2
from falcor2.pxr import Usd, UsdGeom, Sdf
from .mesh_motion_fixtures import add_motion_curve

pytestmark = pytest.mark.nightly


def _options(refinement: int = 0) -> f2.ImportOptions:
    options = f2.ImportOptions()
    options.import_curve_motion = True
    options.curve_refinement_level = refinement
    return options


@pytest.mark.parametrize(
    "curve_type,basis,wrap",
    [
        ("linear", "bezier", "nonperiodic"),
        ("cubic", "bezier", "nonperiodic"),
        ("cubic", "bspline", "nonperiodic"),
        ("cubic", "catmullRom", "nonperiodic"),
        ("cubic", "bspline", "pinned"),
        ("cubic", "catmullRom", "pinned"),
    ],
)
@pytest.mark.parametrize("refinement", [0, 2])
def test_curve_samples_match_static_imports(
    curve_type: str, basis: str, wrap: str, refinement: int
) -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(2)
    source = add_motion_curve(stage, curve_type, basis)
    source.CreateWrapAttr().Set(wrap)
    curve = f2.UsdImporter().import_stage(stage, _options(refinement)).curves[0]
    assert curve.motion_times == [0, 0.125, 0.375, 0.5]
    assert curve.sample_count == 4
    assert curve.motion_interpolation_mode == f2.AnimationInterpolationMode.linear
    np.testing.assert_array_equal(curve.positions, curve.position_sample())
    np.testing.assert_array_equal(curve.radii, curve.radius_sample())
    for sample, time in enumerate((0, 0.25, 0.75, 1)):
        snapshot = Usd.Stage.Open(stage.Flatten())
        static_source = UsdGeom.BasisCurves(snapshot.GetPrimAtPath("/Curves"))
        for source_attr, static_attr in (
            (source.GetPointsAttr(), static_source.GetPointsAttr()),
            (source.GetWidthsAttr(), static_source.GetWidthsAttr()),
        ):
            static_attr.Clear()
            static_attr.Set(source_attr.Get(time))
        options = _options(refinement)
        options.import_curve_motion = False
        reference = f2.UsdImporter().import_stage(snapshot, options).curves[0]
        assert reference.sample_count == 1
        assert not reference.motion_times
        np.testing.assert_allclose(curve.position_sample(sample), reference.positions, atol=1e-6)
        np.testing.assert_allclose(curve.radius_sample(sample), reference.radii, atol=1e-6)
        np.testing.assert_array_equal(curve.indices, reference.indices)
        np.testing.assert_array_equal(curve.texcoords, reference.texcoords)
        if wrap == "pinned":
            controls = np.asarray(source.GetPointsAttr().Get(time))
            input_offset = output_offset = 0
            for count in source.GetCurveVertexCountsAttr().Get():
                output_count = count if refinement == 0 else (count - 1) * (refinement + 1) + 1
                endpoints = curve.position_sample(sample)[
                    [output_offset, output_offset + output_count - 1]
                ]
                np.testing.assert_allclose(
                    endpoints, controls[[input_offset, input_offset + count - 1]], atol=1e-6
                )
                input_offset += count
                output_offset += output_count
            assert output_offset == len(curve.position_sample(sample))
    with pytest.raises(RuntimeError, match="sample index out of bounds"):
        curve.position_sample(curve.sample_count)
    with pytest.raises(RuntimeError, match="sample index out of bounds"):
        curve.radius_sample(curve.sample_count)
    # A retained sample view keeps the curve alive.
    positions = curve.position_sample(1)
    expected = positions.copy()
    del curve
    np.testing.assert_array_equal(positions, expected)


def test_curve_motion_nonpositive_time_rate() -> None:
    stage = Usd.Stage.CreateInMemory()
    add_motion_curve(stage)
    for rate in (0, -24):
        stage.SetTimeCodesPerSecond(rate)
        curve = f2.UsdImporter().import_stage(stage, _options()).curves[0]
        np.testing.assert_allclose(curve.motion_times, np.array([0, 0.25, 0.75, 1]) / 24)


def test_curve_width_only_held_motion() -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(1)
    source = add_motion_curve(stage)
    points = source.GetPointsAttr().Get(0)
    source.GetPointsAttr().Clear()
    source.GetPointsAttr().Set(points)
    stage.SetInterpolationType(Usd.InterpolationTypeHeld)
    curve = f2.UsdImporter().import_stage(stage, _options(2)).curves[0]
    assert curve.motion_times == [0.25, 0.75]
    assert curve.motion_interpolation_mode == f2.AnimationInterpolationMode.step
    # Separately tessellated samples can differ by float32 rounding on ARM64.
    np.testing.assert_allclose(
        curve.position_sample(0), curve.position_sample(1), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(curve.radius_sample(0), 0.05, atol=1e-6)
    np.testing.assert_allclose(curve.radius_sample(1), 0.1, atol=1e-6)


def test_curve_motion_preserves_coincident_reference_edges() -> None:
    stage = Usd.Stage.CreateInMemory()
    source = add_motion_curve(stage)
    points = source.GetPointsAttr().Get(0)
    points[1] = points[0]
    source.GetPointsAttr().Set(points, 0)
    source.GetPointsAttr().Set(points)
    animated = f2.UsdImporter().import_stage(stage, _options()).curves[0]
    ordinary = f2.UsdImporter().import_stage(stage).curves[0]
    assert len(animated.indices) == len(ordinary.indices) + 2
    assert list(animated.indices[:2]) == [0, 1]
    np.testing.assert_array_equal(animated.position_sample(0)[0], animated.position_sample(0)[1])
    assert not np.array_equal(animated.position_sample(3)[0], animated.position_sample(3)[1])


@pytest.mark.parametrize("attribute", ["points", "widths", "curveVertexCounts", "st:indices"])
def test_curve_motion_rejects_changing_layout(attribute: str) -> None:
    stage = Usd.Stage.CreateInMemory()
    source = add_motion_curve(stage)
    if attribute == "st:indices":
        st = UsdGeom.PrimvarsAPI(source).GetPrimvar("st")
        st.SetIndices([0, 1], 0)
        st.SetIndices([0, 1], 1)
        f2.UsdImporter().import_stage(stage, _options())
        st.SetIndices([1, 0], 0.5)
    else:
        attr = source.GetPrim().GetAttribute(attribute)
        if attribute == "curveVertexCounts":
            attr.Set([7, 4], 0)
            attr.Set([7, 4], 1)
            f2.UsdImporter().import_stage(stage, _options())
            attr.Set([4, 7], 0.5)
        elif attribute == "points":
            attr.Set(attr.Get(0)[:-1], 0.5)
        else:
            attr.Set([0.1, 0.2], 0.5)
    with pytest.raises(RuntimeError, match="changes (point count|width count|value count|indices)"):
        f2.UsdImporter().import_stage(stage, _options())


@pytest.mark.parametrize("refinement", [0, 2])
def test_curve_motion_indexed_width_primvar(refinement: int) -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(1)
    source = add_motion_curve(stage)
    points = source.GetPointsAttr().Get(0)
    source.GetPointsAttr().Clear()
    source.GetPointsAttr().Set(points)
    widths = UsdGeom.PrimvarsAPI(source).CreatePrimvar(
        "widths", Sdf.ValueTypeNames.FloatArray, "varying"
    )
    widths.SetIndices([0, 1, 0, 1, 0])
    widths.Set([0.1, 0.4], 2)
    widths.Set([0.2, 0.8], 4)
    curve = f2.UsdImporter().import_stage(stage, _options(refinement)).curves[0]
    # The primvar replaces the schema widths and their different sampling schedule.
    assert curve.motion_times == [2, 4]
    for sample, time in enumerate((2, 4)):
        snapshot = Usd.Stage.Open(stage.Flatten())
        snapshot_widths = snapshot.GetPrimAtPath("/Curves").GetAttribute("primvars:widths")
        snapshot_widths.Clear()
        snapshot_widths.Set(widths.Get(time))
        options = _options(refinement)
        options.import_curve_motion = False
        reference = f2.UsdImporter().import_stage(snapshot, options).curves[0]
        np.testing.assert_allclose(curve.radius_sample(sample), reference.radii, atol=1e-6)
    np.testing.assert_allclose(curve.radius_sample(1), 2 * curve.radius_sample(0), atol=1e-6)
    widths.SetIndices([1, 0, 1, 0, 1], 3)
    widths.SetIndices([0, 1, 0, 1, 0], 4)
    with pytest.raises(RuntimeError, match="changes indices"):
        f2.UsdImporter().import_stage(stage, _options(refinement))


def test_curve_static_uses_earliest_width_sample() -> None:
    stage = Usd.Stage.CreateInMemory()
    curve = UsdGeom.BasisCurves.Define(stage, "/Curve")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr([(0, 0, 0), (1, 0, 0)])
    curve.CreateWidthsAttr().Set([2], 1)
    curve.GetWidthsAttr().Set([4], 2)
    curve.SetWidthsInterpolation("constant")
    imported = f2.UsdImporter().import_stage(stage).curves[0]
    np.testing.assert_allclose(imported.radii, 1)
