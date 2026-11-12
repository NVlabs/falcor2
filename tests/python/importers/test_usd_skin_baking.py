# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import falcor2 as f2
from falcor2.pxr import Gf, Usd, UsdGeom, UsdSkel


def test_baking_skins_imported_normals() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdSkel.Root.Define(stage, "/Rig")
    skeleton = UsdSkel.Skeleton.Define(stage, "/Rig/Skeleton")
    skeleton.CreateJointsAttr(["a", "b"])
    skeleton.CreateBindTransformsAttr([Gf.Matrix4d(1), Gf.Matrix4d(1)])
    skeleton.CreateRestTransformsAttr([Gf.Matrix4d(1), Gf.Matrix4d().SetTranslate((0, 0, 1))])
    mesh = UsdGeom.Mesh.Define(stage, "/Rig/Mesh")
    mesh.CreateSubdivisionSchemeAttr("catmullClark")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateJointIndicesPrimvar(False, 1).Set([0, 0, 1, 0])
    # The source remains untouched; the baking consumer normalizes these weights.
    binding.CreateJointWeightsPrimvar(False, 1).Set([2, 2, 2, 2])
    original = stage.GetRootLayer().ExportToString()
    options = f2.ImportOptions()
    options.bake_poses = True
    options.mesh_refinement_level = 2
    baked = f2.UsdImporter().import_stage(stage, options)
    result = baked.meshes[0]
    assert not baked.skins
    assert result.vertex_count == 4
    assert result.properties["refinement_level"] == 0
    np.testing.assert_allclose(result.positions, [(0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0)])
    # Import generates the base normals. Skinning transforms those normals, just as future GPU skinning will;
    # it does not reconstruct geometric normals from the deformed polygon. These joints only translate.
    np.testing.assert_allclose(result.normals, [(0, 0, 1)] * 4, atol=1e-6)
    assert stage.GetRootLayer().ExportToString() == original


def _animated_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(1)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdSkel.Root.Define(stage, "/Rig")
    skeleton = UsdSkel.Skeleton.Define(stage, "/Rig/Skeleton")
    placement = skeleton.AddTranslateOp()
    placement.Set((0, 10, 0), -2)
    placement.Set((0, 14, 0), 2)
    skeleton.CreateJointsAttr(["a"])
    skeleton.CreateBindTransformsAttr([Gf.Matrix4d(1)])
    skeleton.CreateRestTransformsAttr([Gf.Matrix4d(1)])
    animation = UsdSkel.Animation.Define(stage, "/Rig/Animation")
    animation.CreateJointsAttr(["a"])
    animation.CreateTranslationsAttr().Set([(2, 0, 0)], -2)
    animation.GetTranslationsAttr().Set([(6, 0, 0)], 2)
    animation.CreateRotationsAttr([Gf.Quatf(1)])
    animation.CreateScalesAttr([(1, 1, 1)])
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets(
        [animation.GetPath()]
    )
    mesh = UsdGeom.Mesh.Define(stage, "/Rig/Mesh")
    mesh.AddTranslateOp().Set((3, 0, 0))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateJointIndicesPrimvar(False, 1).Set([0, 0, 0])
    binding.CreateJointWeightsPrimvar(False, 1).Set([1, 1, 1])
    return stage


@pytest.mark.parametrize("time_seconds", [-2, 0, 0.5, 2])
def test_usd_baking_uses_retained_animation(time_seconds: float) -> None:
    stage = _animated_stage()
    options = f2.ImportOptions()
    options.usd_import_skinning = True
    scene = f2.UsdImporter().import_stage(stage, options)
    source_positions = np.array(scene.meshes[0].positions)
    # The source is no longer available. Only the retained channels, joints, and binding can supply this pose.
    stage.RemovePrim("/Rig")
    scene.bake_pose(time_seconds)
    np.testing.assert_allclose(
        scene.meshes[0].positions,
        source_positions + [1 + time_seconds, 12 + time_seconds, 0],
        atol=1e-6,
    )
    assert not scene.skins
    assert scene.animation.channels


def test_usd_convenience_bake_matches_initial_viewing_time() -> None:
    stage = _animated_stage()
    for source_rate in (2, 0):
        stage.SetTimeCodesPerSecond(source_rate)
        stage.GetRootLayer().ClearStartTimeCode()
        stage.GetRootLayer().ClearEndTimeCode()
        for time in (0, 1):
            if time:
                stage.SetStartTimeCode(time)
                stage.SetEndTimeCode(600)
            original = stage.GetRootLayer().ExportToString()
            options = f2.ImportOptions()
            options.usd_import_skinning = True
            retained = f2.UsdImporter().import_stage(stage, options)
            rate = stage.GetTimeCodesPerSecond() if stage.GetTimeCodesPerSecond() > 0 else 24
            for channel in retained.animation.channels:
                assert np.isfinite(channel.times).all()
            retained.bake_pose(time / rate)
            options.bake_poses = True
            baked = f2.UsdImporter().import_stage(stage, options)
            np.testing.assert_array_equal(baked.meshes[0].positions, retained.meshes[0].positions)
            np.testing.assert_array_equal(baked.meshes[0].normals, retained.meshes[0].normals)
            assert len(baked.animation.channels) == len(retained.animation.channels)
            for actual, expected in zip(baked.nodes, retained.nodes):
                np.testing.assert_array_equal(
                    actual.transform.to_numpy(), expected.transform.to_numpy()
                )
            assert stage.GetRootLayer().ExportToString() == original


def test_usd_mesh_bindings_share_joint_nodes() -> None:
    stage = _animated_stage()
    second = UsdGeom.Mesh.Define(stage, "/Rig/OtherMesh")
    second.GetPrim().GetReferences().AddInternalReference("/Rig/Mesh")
    UsdSkel.BindingAPI.Apply(second.GetPrim()).CreateGeomBindTransformAttr(
        Gf.Matrix4d().SetTranslate((10, 0, 0))
    )
    options = f2.ImportOptions()
    options.usd_import_skinning = True
    scene = f2.UsdImporter().import_stage(stage, options)
    assert len(scene.skins) == 2
    assert scene.skins[0].joints == scene.skins[1].joints
    assert len(scene.animation.channels) == 4  # Skeleton placement and one joint's TRS channels.
    scene.bake_pose(0.5)
    first_mesh = next(mesh for mesh in scene.meshes if str(mesh.name) == "/Rig/Mesh")
    second_mesh = next(mesh for mesh in scene.meshes if str(mesh.name) == "/Rig/OtherMesh")
    np.testing.assert_allclose(
        second_mesh.positions, np.asarray(first_mesh.positions) + [10, 0, 0], atol=1e-6
    )
