# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import falcor2 as f2
import falcor2.testing.helpers as helpers
import pytest
import slangpy as spy


def _material(name: str) -> str:
    return f"""def Material "{name}" {{
    token outputs:surface.connect = </{name}/Shader.outputs:surface>
    def Shader "Shader" {{
        uniform token info:id = "UsdPreviewSurface"
        token outputs:surface
    }}
}}
"""


def _mesh(name: str, double_sided: bool, binding: str, subset: bool = False) -> str:
    binding_text = f"rel material:binding = </{binding}>"
    if subset:
        binding_text = f"""def GeomSubset "Subset" {{
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [0]
            {binding_text}
        }}"""
    return f"""def Mesh "{name}" {{
    uniform bool doubleSided = {int(double_sided)}
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    {binding_text}
}}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_usd_double_sided_material(tmp_path: Path, device: spy.Device) -> None:
    path = tmp_path / "double_sided.usda"
    path.write_text(
        "#usda 1.0\n"
        + _material("Shared")
        + _material("Single")
        + _mesh("Front", False, "Shared")
        + _mesh("Back", True, "Shared", subset=True)
        + _mesh("SingleMesh", False, "Single"),
        encoding="utf-8",
    )
    imported = f2.UsdImporter().load_scene(path)
    materials = {material.name: material for material in imported.materials}
    assert len(imported.materials) == 2
    assert materials["/Shared"].params["double_sided"]
    assert "double_sided" not in materials["/Single"].params

    scene = f2.Scene.from_importer_scene(device, imported)
    materials = {material.name: material for material in scene.materials}
    assert len(scene.materials) == 2
    assert materials["/Shared"].double_sided
    assert not materials["/Shared"].enable_backface_culling
    assert not materials["/Single"].double_sided
