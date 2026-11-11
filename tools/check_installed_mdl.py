# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check bundled OmniPBR resolution. Run with the installed interpreter and -I."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

import falcor2 as f2
from falcor2.pxr import Sdf, Usd, UsdGeom, UsdShade
import slangpy as spy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["d3d12", "vulkan"], default="d3d12" if os.name == "nt" else "vulkan"
    )
    args = parser.parse_args()
    os.environ.pop("MDL_SYSTEM_PATH", None)
    os.environ.pop("MDL_USER_PATH", None)
    if not Path(f2.falcor2_ext.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Expected the pip-installed Falcor2 package")

    device = f2.create_device(spy.DeviceType[args.backend])
    with tempfile.TemporaryDirectory(prefix="falcor-installed-mdl-") as directory:
        stage = Usd.Stage.CreateInMemory()
        material = UsdShade.Material.Define(stage, "/Material")
        shader = UsdShade.Shader.Define(stage, "/Material/Shader")
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        material.CreateSurfaceOutput("mdl").ConnectToSource(
            shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        )
        mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        path = Path(directory) / "fixture.usda"
        stage.GetRootLayer().Export(str(path))

        scene = f2.Scene.load(device, path)
        if len(scene.materials) != 1 or not isinstance(scene.materials[0], f2.MDLMaterial):
            raise RuntimeError("OmniPBR did not resolve to an MDL material")
        scene.update()
    device.close()
    print("Bundled OmniPBR resolved successfully")


if __name__ == "__main__":
    main()
