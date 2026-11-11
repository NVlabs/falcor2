# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analytic USD fixtures for mesh-motion import and static-geometry tests."""

import numpy as np
import falcor2 as f2
from falcor2.pxr import Usd, UsdGeom, Sdf


def mesh_motion_options() -> f2.ImportOptions:
    options = f2.ImportOptions()
    options.import_mesh_motion = True
    return options


def seam_stage() -> Usd.Stage:
    """Two triangles sharing an initially equal corner, with mirrored UVs and a rotating pose."""
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(1)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    mesh = UsdGeom.Mesh.Define(stage, "/Seam")
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
    # Expanded version of test_vertexstreams' shared-origin two-triangle fixture.
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float32
    )
    mesh.CreatePointsAttr(positions.tolist()).Set(positions.tolist(), 0)
    rotated = positions[:, [1, 0, 2]].copy()
    rotated[:, 0] *= -1
    rotated[3:, 2] += 0.25
    mesh.GetPointsAttr().Set(rotated.tolist(), 1)
    mesh.CreateNormalsAttr([(0, 0, 1)] * 6)
    mesh.SetNormalsInterpolation("vertex")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    st.Set([(0, 0), (1, 0), (0, 1), (0, 0), (0, 1), (1, 0)])
    return stage


def import_seam() -> f2.ImporterScene:
    return f2.UsdImporter().import_stage(seam_stage(), mesh_motion_options())


def add_motion_curve(
    stage: Usd.Stage, curve_type: str = "cubic", basis: str = "bezier"
) -> UsdGeom.BasisCurves:
    """Two strands with independently sampled positions and widths."""
    curve = UsdGeom.BasisCurves.Define(stage, "/Curves")
    curve.CreateTypeAttr(curve_type)
    curve.CreateBasisAttr(basis)
    curve.CreateWrapAttr("nonperiodic")
    curve.CreateCurveVertexCountsAttr([7, 4])
    positions = np.array(
        [
            (-0.8, 0, 0),
            (-0.5, 0.6, 0),
            (0.5, 0.6, 0),
            (0.8, 0, 0),
            (1.1, -0.6, 0),
            (1.6, -0.6, 0),
            (2, 0, 0),
            (-0.8, -1, 0),
            (-0.5, -0.4, 0),
            (0.5, -0.4, 0),
            (0.8, -1, 0),
        ],
        dtype=np.float32,
    )
    curve.CreatePointsAttr(positions.tolist()).Set(positions.tolist(), 0)
    positions[:, 2] = np.linspace(0, 0.5, len(positions))
    curve.GetPointsAttr().Set(positions.tolist(), 1)
    curve.CreateWidthsAttr([0.1]).Set([0.1], 0.25)
    curve.GetWidthsAttr().Set([0.2], 0.75)
    curve.SetWidthsInterpolation("constant")
    st = UsdGeom.PrimvarsAPI(curve).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "uniform"
    )
    st.Set([(0.2, 0.8), (0.7, 0.3)])
    return curve
