# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
from PIL import Image

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper


def _write_mixed_texture_scene(tmp_path: Path) -> tuple[Path, Path, bytes]:
    image_path = tmp_path / "grey.png"
    Image.new("L", (2, 2), 128).save(image_path)
    image_bytes = image_path.read_bytes()
    document = {
        "asset": {"version": "2.0"},
        "images": [
            {"uri": image_path.name},
            {"uri": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")},
        ],
        "textures": [{"source": 0}, {"source": 1}],
        "materials": [
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": index},
                    "metallicRoughnessTexture": {"index": index},
                },
            }
            for index, name in enumerate(("external", "embedded"))
        ],
    }
    scene_path = tmp_path / "mixed_sources.gltf"
    scene_path.write_text(json.dumps(document), encoding="utf-8")
    return scene_path, image_path, image_bytes


def test_gltf_external_image_remains_path_backed(tmp_path: Path) -> None:
    scene_path, image_path, image_bytes = _write_mixed_texture_scene(tmp_path)
    device = helpers.get_device(helpers.DEFAULT_DEVICE_TYPES[0])
    scene = f2.Scene.load(device, scene_path)
    external, embedded = scene.materials

    assert external.base_color_texture_path == image_path
    assert external.base_color_texture_bytes is None
    assert external.metallic_roughness_texture_path == image_path
    assert external.metallic_roughness_texture_bytes is None
    assert embedded.base_color_texture_bytes.data == image_bytes
    assert embedded.metallic_roughness_texture_bytes.data == image_bytes
    assert embedded.base_color_texture is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_mdl_file_and_embedded_linear_textures_match(
    tmp_path: Path, device_type: spy.DeviceType
) -> None:
    scene_path, image_path, image_bytes = _write_mixed_texture_scene(tmp_path)
    options = f2.ImportOptions()
    options.gltf_material_mode = f2.GLTFMaterialMode.mdl
    device = helpers.get_device(device_type)
    scene = f2.Scene.load(device, scene_path, import_options=options)
    external, embedded = scene.materials
    assert isinstance(external, f2.MDLMaterial)
    assert isinstance(embedded, f2.MDLMaterial)
    assert external["metallic_roughness_texture_map"] == str(image_path)
    assert external["metallic_roughness_texture_map:texture_bytes"] is None
    assert embedded["metallic_roughness_texture_map:texture_bytes"].data == image_bytes
    assert not external["metallic_roughness_texture_map:srgb"]
    assert not embedded["metallic_roughness_texture_map:srgb"]

    scene.update()
    external_handles = external.build_texture_list()
    embedded_handles = embedded.build_texture_list()
    assert any(handle.path == image_path for handle in external_handles)
    assert any(handle.path is None for handle in embedded_handles)

    helper = SceneShaderHelper(device)
    module, _ = helper.prepare(scene, "falcor2.tools.materials.scene_sampling_tools")
    collect = module["material_collect_properties_simple"].as_func().write(helper.bind_scene)
    uvs = np.array([[0.5, 0.5]], dtype=np.float32)
    wi = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    values = []
    for material in (external, embedded):
        result = collect(material, uvs, wi)
        cursor = result.cursor()
        cursor.load()
        values.append(cursor[0].read())
    # MDL's material guide exposes sampled albedo and roughness, but does not
    # populate the glTF-specific metallic guide field.
    for key in (
        "material_color",
        "diffuse_reflection_albedo",
        "specular_reflection_albedo",
        "roughness",
    ):
        np.testing.assert_allclose(values[0][key], values[1][key], atol=2e-4)
    assert values[0]["material_color"].x > 0.01
