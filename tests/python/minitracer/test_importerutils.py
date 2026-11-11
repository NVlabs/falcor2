# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest
import slangpy as spy
from PIL import Image

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.minitracer.importerutils import load_full_materials
from falcor2.minitracer.material import MaterialTexture
from falcor2.minitracer.scene import Scene


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    ("texture_info", "expected_offset", "expected_scale", "expected_rotation", "expected_texcoord"),
    [
        pytest.param({}, (0.0, 0.0), (1.0, 1.0), 0.0, 0, id="defaults"),
        pytest.param({"texCoord": 1}, (0.0, 0.0), (1.0, 1.0), 0.0, 1, id="uv1"),
        pytest.param(
            {
                "texCoord": 0,
                "extensions": {
                    "KHR_texture_transform": {
                        "offset": [0.25, -0.5],
                        "scale": [2.0, 3.0],
                        "rotation": 0.7,
                        "texCoord": 1,
                    }
                },
            },
            (0.25, -0.5),
            (2.0, 3.0),
            0.7,
            1,
            id="transform-uv-override",
        ),
    ],
)
def test_gltf_material_texture_coordinates(
    device: spy.Device,
    tmp_path: Path,
    texture_info: dict[str, Any],
    expected_offset: tuple[float, float],
    expected_scale: tuple[float, float],
    expected_rotation: float,
    expected_texcoord: int,
) -> None:
    texture_path = tmp_path / "texture.png"
    Image.new("RGBA", (1, 1), (128, 128, 255, 255)).save(texture_path)
    scene_path = tmp_path / "material.gltf"
    scene_path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "extensionsUsed": ["KHR_texture_transform"],
                "images": [{"uri": texture_path.name}],
                "textures": [{"source": 0}],
                "materials": [
                    {
                        "pbrMetallicRoughness": {
                            "baseColorTexture": {"index": 0, **texture_info},
                        },
                        "normalTexture": {"index": 0, "scale": 0.6, **texture_info},
                    }
                ],
                "nodes": [],
                "scenes": [{"nodes": []}],
                "scene": 0,
            }
        ),
        encoding="utf-8",
    )

    importer_scene = f2.GltfImporter().load_scene(scene_path)
    scene = Scene(device)
    (material,) = load_full_materials(scene, importer_scene)

    for texture in (material.albedo, material.normal):
        assert isinstance(texture, MaterialTexture)
        assert (texture.offset.x, texture.offset.y) == pytest.approx(expected_offset)
        assert (texture.scale.x, texture.scale.y) == pytest.approx(expected_scale)
        assert texture.rotation == pytest.approx(expected_rotation)
        assert texture.texcoord == expected_texcoord
    assert material.normal_map_strength == pytest.approx(0.6)
