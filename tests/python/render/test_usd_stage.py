# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers


def test_usd_stage_stubs_import_pxr() -> None:
    stub_path = Path(f2.__file__).with_name("__init__.pyi")
    tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))

    imports_pxr = any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module is None
        and any(alias.name == "pxr" for alias in node.names)
        for node in tree.body
    )
    assert imports_pxr

    scene_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scene"
    )
    from_stage = next(
        node
        for node in scene_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "from_stage"
    )
    usd_stage = next(arg for arg in from_stage.args.args if arg.arg == "usd_stage")
    assert ast.unparse(usd_stage.annotation) == "pxr.Usd.Stage"
    import_options = next(arg for arg in from_stage.args.args if arg.arg == "import_options")
    assert ast.unparse(import_options.annotation) == "ImportOptions | None"

    importer_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UsdImporter"
    )
    import_stage = next(
        node
        for node in importer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "import_stage"
    )
    stage = next(arg for arg in import_stage.args.args if arg.arg == "stage")
    assert ast.unparse(stage.annotation) == "pxr.Usd.Stage"


@pytest.mark.parametrize("bake_poses", [False, True])
def test_usd_importer_import_stage(bake_poses: bool) -> None:
    from falcor2.pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Triangle")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    options = f2.ImportOptions()
    options.bake_poses = bake_poses
    importer_scene = f2.UsdImporter().import_stage(stage, options)

    assert len(importer_scene.meshes) == 1


@pytest.mark.parametrize("refinement", [0, 2])
def test_usd_bake_frame_zero(tmp_path: Path, refinement: int) -> None:
    from falcor2.pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdSkel.Root.Define(stage, "/Character")
    root.AddTranslateOp().Set((10, 0, 0))
    skeleton = UsdSkel.Skeleton.Define(stage, "/Character/Skeleton")
    skeleton.CreateJointsAttr(["joint"])
    skeleton.CreateBindTransformsAttr([Gf.Matrix4d().SetTranslate((1, 0, 0))])
    skeleton.CreateRestTransformsAttr([Gf.Matrix4d().SetTranslate((1, 0, 0))])
    animation = UsdSkel.Animation.Define(stage, "/Character/Animation")
    animation.CreateJointsAttr(["joint"])
    # No sample at zero: the baked pose must clamp to the first animated pose, not rest.
    animation.CreateTranslationsAttr().Set([(3, 0, 0)], 1)
    animation.GetTranslationsAttr().Set([(9, 0, 0)], 10)
    animation.CreateRotationsAttr([Gf.Quatf(1)])
    animation.CreateScalesAttr([(1, 1, 1)])
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets(
        [animation.GetPath()]
    )

    mesh = UsdGeom.Mesh.Define(stage, "/Character/Body")
    mesh.AddTranslateOp().Set((2, 0, 0))
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(refinement)
    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateGeomBindTransformAttr(Gf.Matrix4d().SetTranslate((2, 0, 0)))
    binding.CreateJointIndicesPrimvar(False, 1).Set([0, 0, 0, 0])
    binding.CreateJointWeightsPrimvar(False, 1).Set([1, 1, 1, 1])

    unskinned = UsdGeom.Mesh.Define(stage, "/Character/Unskinned")
    unskinned.CreatePointsAttr(mesh.GetPointsAttr().Get())
    unskinned.CreateFaceVertexCountsAttr([4])
    unskinned.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    unskinned.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(2)

    fur = UsdGeom.BasisCurves.Define(stage, "/Character/Fur")
    fur.CreateTypeAttr("linear")
    fur.CreateCurveVertexCountsAttr([2])
    fur.CreateWidthsAttr([0.1])
    fur.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    fur.CreatePointsAttr([(100, 0, 0), (101, 0, 0)])
    fur.GetPointsAttr().Set([(4, 0, 0), (5, 0, 0)], 1)
    fur.GetPointsAttr().Set([(10, 0, 0), (11, 0, 0)], 10)
    camera = UsdGeom.Camera.Define(stage, "/Character/Camera")
    camera.AddTranslateOp().Set((4, 0, 5), 0)
    camera.GetOrderedXformOps()[0].Set((10, 0, 5), 10)

    source = stage.GetRootLayer().ExportToString()
    session = stage.GetSessionLayer().ExportToString()
    importer = f2.UsdImporter()
    raw = importer.import_stage(stage)
    raw_meshes = {str(mesh.name): mesh for mesh in raw.meshes}
    assert (raw_meshes["/Character/Body"].vertex_count > 4) == (refinement > 0)
    assert raw_meshes["/Character/Unskinned"].vertex_count > 4
    assert len(raw.animation.channels) > 0

    options = f2.ImportOptions()
    options.bake_poses = True
    path = tmp_path / "pose.usda"
    stage.GetRootLayer().Export(str(path))
    file_stage = Usd.Stage.Open(str(path))
    file_source = file_stage.GetRootLayer().ExportToString()
    file_session = file_stage.GetSessionLayer().ExportToString()
    cache = Usd.StageCache()
    cache.Insert(file_stage)
    with Usd.StageCacheContext(cache):
        baked_file = importer.load_scene(path, options)
    for baked in (importer.import_stage(stage, options), baked_file):
        meshes = {str(mesh.name): mesh for mesh in baked.meshes}
        assert meshes["/Character/Body"].vertex_count == 4
        np.testing.assert_array_equal(
            meshes["/Character/Unskinned"].positions, raw_meshes["/Character/Unskinned"].positions
        )
        # A skin pose bake retains ordinary animation; it no longer freezes the entire USD stage.
        assert len(baked.animation.channels) > 0
        # In skeleton space: animated joint 3 - bind joint 1 + geometry bind 2 = 4.
        # Mesh-local points subtract the mesh translation (2); the parent adds 10.
        np.testing.assert_allclose(
            meshes["/Character/Body"].positions,
            np.asarray(mesh.GetPointsAttr().Get()) + [2, 0, 0],
            atol=1e-6,
        )
        np.testing.assert_array_equal(baked.curves[0].positions, raw.curves[0].positions)
        camera_node = next(node for node in baked.nodes if node.camera_index >= 0)
        np.testing.assert_allclose(camera_node.transform.to_numpy()[:3, 3], [4, 0, 5])
    assert stage.GetRootLayer().ExportToString() == source
    assert stage.GetSessionLayer().ExportToString() == session
    assert file_stage.GetRootLayer().ExportToString() == file_source
    assert file_stage.GetSessionLayer().ExportToString() == file_session


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_usd_texcoord_primvar_mapping_stage_entry_points(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    from falcor2.pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Triangle")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lightmap = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "lightmap", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    lightmap.Set([(4, 1), (5, 1), (4, 2)])

    options = f2.ImportOptions()
    names = list(options.usd_texcoord_primvar_names)
    names[0] = "lightmap"
    options.usd_texcoord_primvar_names = names

    imported = f2.UsdImporter().import_stage(stage, options)
    uv1 = imported.meshes[0].texcoords(1)
    assert uv1 is not None
    np.testing.assert_array_equal(uv1[0], [4.0, 1.0])

    scene = f2.Scene.from_stage(device, stage, options)
    geometry = scene.geometries[0]
    uv1 = next(
        attribute
        for attribute in geometry.vertex_attributes(0)
        if attribute.semantic == f2.MeshAttributeSemantic.tex_coord and attribute.index == 1
    )
    assert uv1.data[0, 0] == 4.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_scene_from_usd_stage(device_type: spy.DeviceType, tmp_path: Path) -> None:
    from falcor2.pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(2)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    transform = UsdGeom.Xform.Define(stage, "/StageRoot").AddTranslateOp()
    transform.Set((10, 0, 0), 10)
    transform.Set((30, 0, 0), 30)
    mesh = UsdGeom.Mesh.Define(stage, "/StageRoot/Triangle")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    device = helpers.get_device(device_type)
    for authored in (False, True):
        if authored:
            stage.SetStartTimeCode(20)
            stage.SetEndTimeCode(24)
        expected_start, expected_end = (10, 12) if authored else (5, 15)
        imported = f2.UsdImporter().import_stage(stage)
        assert imported.animation.key_range().start == 5
        assert imported.animation.key_range().end == 15
        assert imported.animation_range().start == expected_start
        assert imported.animation_range().end == expected_end
        path = tmp_path / "animated.usda"
        stage.GetRootLayer().Export(str(path))
        for scene in (f2.Scene.from_stage(device, stage), f2.Scene.load(device, path)):
            entity = scene.entities.find(name="/StageRoot")
            assert entity is not None
            assert len(scene.geometries) == 1
            assert scene.animation_range.start == expected_start
            assert scene.animation_range.end == expected_end
            assert scene.time == expected_start
            scene.update()
            assert entity.transform.translation.x == pytest.approx(2 * expected_start)
            scene.time = expected_start + 1
            scene.update()
            assert entity.transform.translation.x == pytest.approx(2 * (expected_start + 1))
            scene.append(imported)
            assert scene.time == expected_start + 1
            scene.update()
            assert entity.transform.translation.x == pytest.approx(2 * (expected_start + 1))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_scene_from_usd_stage_applies_import_options(device_type: spy.DeviceType) -> None:
    from falcor2.pxr import Usd, UsdLux

    stage = Usd.Stage.CreateInMemory()
    UsdLux.RectLight.Define(stage, "/RectLight")

    options = f2.ImportOptions()
    options.force_rectangle_lights_to_geometry = True

    device = helpers.get_device(device_type)
    scene = f2.Scene.from_stage(device, stage, options)

    assert len(scene.geometries) == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_scene_from_usd_stage_rejects_other_objects(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    with pytest.raises(TypeError, match="falcor2.pxr.Usd.Stage"):
        f2.Scene.from_stage(device, object())

    if importlib.util.find_spec("pxr") is not None:
        from pxr import Usd as system_usd

        with pytest.raises(TypeError, match="falcor2.pxr.Usd.Stage"):
            f2.Scene.from_stage(device, system_usd.Stage.CreateInMemory())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_scene_from_usd_stage_does_not_modify_stage(device_type: spy.DeviceType) -> None:
    from falcor2.pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    root_before = stage.GetRootLayer().ExportToString()
    session_before = stage.GetSessionLayer().ExportToString()

    device = helpers.get_device(device_type)
    f2.Scene.from_stage(device, stage)

    assert stage.GetRootLayer().ExportToString() == root_before
    assert stage.GetSessionLayer().ExportToString() == session_before


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_scene_from_usd_stage_preprocesses_materialx(device_type: spy.DeviceType) -> None:
    from falcor2.pxr import Usd

    scene_path = (
        Path(__file__).parents[2]
        / "native"
        / "usd_importer"
        / "test_scope_materials_mtlx_openpbr.usda"
    )
    stage = Usd.Stage.Open(str(scene_path))
    assert stage is not None

    device = helpers.get_device(device_type)
    file_scene = f2.Scene.load(device, scene_path)
    stage_scene = f2.Scene.from_stage(device, stage)

    assert len(stage_scene.materials) == len(file_scene.materials)
    assert len(stage_scene.materials) > 0
