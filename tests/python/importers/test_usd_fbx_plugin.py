# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration checks for development builds with the optional USD FBX plugin."""

import ctypes
import json
import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import falcor2 as f2
import numpy as np
import pytest
import slangpy as spy
from falcor2.pxr import Ar, Plug, Sdf, Usd, UsdGeom, UsdShade


@pytest.fixture(scope="module")
def adobe_plugin() -> Iterator[Path]:
    marker = Path(f2.__file__).parent / ".build_dir"
    if not marker.is_file():
        pytest.skip("The USD FBX plugin is only available in development builds")
    runtime = Path(marker.read_text(encoding="utf-8").strip())
    config_path = runtime / "usd_fbx_config.json"
    if not config_path.is_file():
        pytest.skip("No USD FBX configuration for the active Falcor2 build")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["usd_fbx_prefix"] is None:
        pytest.skip("Falcor2 was built without the USD FBX plugin")
    prefix = Path(config["usd_fbx_prefix"])
    registry = prefix / "share/usd-fileformat-plugins/plugin/usdFbx/resources/plugInfo.json"
    assert registry.is_file(), registry
    with ExitStack() as stack:
        if os.name == "nt":
            stack.enter_context(os.add_dll_directory(str(prefix / "bin")))
            # USD uses LoadLibrary; preload with Python's dependency search flags.
            plugin_dll = ctypes.WinDLL(str(prefix / "bin/falcor2_adobe_usdFbx.dll"))
        else:
            suffix = ".dylib" if sys.platform == "darwin" else ".so"
            plugin_dll = ctypes.CDLL(str(prefix / f"lib/libfalcor2_adobe_usdFbx{suffix}"))
        assert plugin_dll
        Plug.Registry().RegisterPlugins(str(registry))
        plugin = Plug.Registry().GetPluginWithName("usdFbx_plugin")
        assert plugin is not None
        assert plugin.Load()
        assert Sdf.FileFormat.FindByExtension("fbx") is not None
        yield prefix


def test_adobe_fbx_cube(adobe_plugin: Path) -> None:
    asset = adobe_plugin / "share/usd-fileformat-plugins/tests/SanityCube.fbx"
    stage = Usd.Stage.Open(str(asset))
    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 1
    assert len(meshes[0].GetPointsAttr().Get()) == 8
    imported = f2.UsdImporter().load_scene(asset)
    assert len(imported.meshes) == 1
    assert imported.meshes[0].vertex_count >= 8
    assert np.isfinite(imported.meshes[0].positions).all()


def test_adobe_fbx_embedded_texture(adobe_plugin: Path, tmp_path: Path) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Y")
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Quad")
    mesh.CreatePointsAttr([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    uv.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    material = UsdShade.Material.Define(stage, "/Root/Material")
    surface = UsdShade.Shader.Define(stage, "/Root/Material/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    texture = UsdShade.Shader.Define(stage, "/Root/Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    reader = UsdShade.Shader.Define(stage, "/Root/Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    texture_path = tmp_path / "source.png"
    spy.Bitmap(np.full((2, 2, 3), 128, dtype=np.uint8)).write(texture_path)
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    asset = tmp_path / "textured.fbx"
    assert stage.Export(str(asset), args={"embedImages": "true"})
    texture_path.unlink()

    reopened = Usd.Stage.Open(str(asset))
    packaged = []
    for prim in reopened.Traverse():
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                resolved = value.resolvedPath or str(Ar.GetResolver().Resolve(value.path))
                if Ar.IsPackageRelativePath(resolved):
                    packaged.append(resolved)
    assert packaged, "The exported FBX should expose its embedded image through ArResolver"
    for path in packaged:
        image = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(path))
        assert image is not None
        assert image.GetSize() > 0
    imported = f2.UsdImporter().load_scene(asset)
    assert len(imported.meshes) == 1
    assert imported.materials
    assert any(
        bytes(image.data).startswith(b"\x89PNG\r\n\x1a\n") for image in imported.assets
    ), "Falcor2 should retain the embedded PNG after the original file is removed"
