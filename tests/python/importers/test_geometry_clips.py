# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""USD remains the oracle for clip composition; native storage reuses complete samples."""

from pathlib import Path

import numpy as np
import pytest

import falcor2 as f2
from falcor2.pxr import Sdf, Usd, UsdGeom

pytestmark = pytest.mark.nightly


def _stage(tmp_path: Path, clip_format: str = "usda") -> Usd.Stage:
    # Both clips use the same source timestamps, but contain different geometry.
    for clip in range(2):
        source = Usd.Stage.CreateNew(str(tmp_path / f"clip{clip}.{clip_format}"))
        mesh = UsdGeom.Mesh.Define(source, "/Asset/Mesh")
        curve = UsdGeom.BasisCurves.Define(source, "/Asset/Curve")
        for time, height in enumerate([0, 2 + clip, 0]):
            points = [(0, 0, clip * 3), (1, height, clip * 3), (2, 1, clip * 3)]
            mesh.CreatePointsAttr().Set(points, time)
            curve.CreatePointsAttr().Set(points, time)
            # Equal positions alone do not identify a complete curve sample.
            curve.CreateWidthsAttr().Set([0.2 + time * 0.2], time)
        source.GetRootLayer().Save()

    manifest = Usd.Stage.CreateNew(str(tmp_path / "manifest.usda"))
    UsdGeom.Mesh.Define(manifest, "/Asset/Mesh").CreatePointsAttr()
    curve = UsdGeom.BasisCurves.Define(manifest, "/Asset/Curve")
    curve.CreatePointsAttr()
    curve.CreateWidthsAttr()
    manifest.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    stage.SetTimeCodesPerSecond(1)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    curve = UsdGeom.BasisCurves.Define(stage, "/Asset/Curve")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([3])
    curve.SetWidthsInterpolation("constant")
    clips = Usd.ClipsAPI(root)
    clips.SetClipAssetPaths([Sdf.AssetPath(f"clip{i}.{clip_format}") for i in range(2)])
    clips.SetClipManifestAssetPath(Sdf.AssetPath("manifest.usda"))
    clips.SetClipPrimPath("/Asset")
    clips.SetClipActive([(0, 0), (2, 1), (4, 0)])
    # Two source frames per second: peaks at .5, 1.5, ... must not disappear.
    clips.SetClipTimes(
        [(0, 0), (1, 2), (1, 0), (2, 2), (2, 0), (3, 2), (3, 0), (4, 2), (4, 0), (5, 2)]
    )
    return stage


def _import(stage: Usd.Stage) -> f2.ImporterScene:
    options = f2.ImportOptions()
    options.import_mesh_motion = True
    options.import_curve_motion = True
    options.curve_refinement_level = 0
    return f2.UsdImporter().import_stage(stage, options)


def _evaluate(
    times: list[float], indices: list[int], samples: list[np.ndarray], time: float, held: bool
) -> np.ndarray:
    right = int(np.searchsorted(times, time, side="right"))
    if right == 0:
        return samples[indices[0]]
    if right == len(times) or held or time == times[right - 1]:
        return samples[indices[right - 1]]
    weight = (time - times[right - 1]) / (times[right] - times[right - 1])
    return samples[indices[right - 1]] * (1 - weight) + samples[indices[right]] * weight


@pytest.mark.parametrize("clip_format", ["usda", "usdc"])
@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_reused_clip_samples(tmp_path: Path, held: bool, reverse: bool, clip_format: str) -> None:
    stage = _stage(tmp_path, clip_format)
    if reverse:
        clips = Usd.ClipsAPI(stage.GetPrimAtPath("/Asset"))
        times = list(clips.GetClipTimes())
        times[-2:] = [(4, 2), (5, 0)]
        clips.SetClipTimes(times)
    if held:
        stage.SetInterpolationType(Usd.InterpolationTypeHeld)
    imported = _import(stage)
    mesh, curve = imported.meshes[0], imported.curves[0]
    assert len(set(mesh.motion_sample_indices)) < len(mesh.motion_times)
    assert 5 <= curve.sample_count <= 8
    assert len(curve.motion_times) > curve.sample_count
    assert len(curve.motion_times) == len(curve.motion_sample_indices)
    assert any(a == b for a, b in zip(curve.motion_times, curve.motion_times[1:]))
    positions = [
        mesh.find_stream(f2.ImporterSemantic.position, i)
        for i in range(max(mesh.motion_sample_indices) + 1)
    ]
    curve_positions = [curve.position_sample(i) for i in range(curve.sample_count)]
    radii = [curve.radius_sample(i) for i in range(curve.sample_count)]
    source_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Mesh"))
    source_curve = UsdGeom.BasisCurves(stage.GetPrimAtPath("/Asset/Curve"))
    queries = list(np.arange(-0.25, 5.5, 0.125)) + [
        boundary + offset for boundary in (1, 2, 3, 4) for offset in (-1e-5, 0, 1e-5)
    ]
    np.random.default_rng(42).shuffle(queries)
    for time in queries:
        for geometry, samples, expected in [
            (mesh, positions, source_mesh.GetPointsAttr().Get(time)),
            (curve, curve_positions, source_curve.GetPointsAttr().Get(time)),
            (curve, radii, np.full(3, source_curve.GetWidthsAttr().Get(time)[0] * 0.5)),
        ]:
            np.testing.assert_allclose(
                _evaluate(
                    geometry.motion_times, geometry.motion_sample_indices, samples, time, held
                ),
                expected,
                atol=2e-6,
                err_msg=f"time={time}",
            )


def test_clip_topology_change_rejected(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Mesh"))
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2], 0)
    mesh.GetFaceVertexIndicesAttr().Set([0, 2, 1], 1)
    with pytest.raises(RuntimeError, match="changes indices"):
        _import(stage)


def test_clip_missing_points_rejected(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    source = Usd.Stage.Open(str(tmp_path / "clip1.usda"))
    UsdGeom.Mesh(source.GetPrimAtPath("/Asset/Mesh")).GetPointsAttr().Set(Sdf.ValueBlock(), 1)
    with pytest.raises(RuntimeError, match="missing or blocked"):
        _import(stage)
