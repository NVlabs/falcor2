# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper


DATA_DIR = Path(__file__).resolve().parents[3] / "data"


@pytest.mark.parametrize(
    "device_type",
    [
        device_type
        for device_type in helpers.DEFAULT_DEVICE_TYPES
        if device_type != spy.DeviceType.cuda
    ],
)
def test_gltf_materialx_embedded_texture_uses_disk_cache(
    tmp_path: Path, device_type: spy.DeviceType
) -> None:
    image_bytes = (DATA_DIR / "assets/textures/test_texture_manager/regular_blue.png").read_bytes()
    image_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    document = {
        "asset": {"version": "2.0"},
        "images": [{"uri": image_uri}],
        "textures": [{"source": 0}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                }
            }
        ],
    }
    path = tmp_path / "embedded_materialx.gltf"
    path.write_text(json.dumps(document), encoding="utf-8")

    options = f2.ImportOptions()
    options.gltf_material_mode = f2.GLTFMaterialMode.materialx
    device = helpers.get_device(device_type)
    scene = f2.Scene.load(device, path, import_options=options)
    material = scene.materials[0]
    assert isinstance(material, f2.MaterialXMaterial)

    # Leave mtlx_layering_mode unset to exercise the build's default backend.
    scene.update()
    handles = material.build_texture_list()
    assert len(handles) == 1
    assert handles[0].texture is not None
    assert handles[0].path is not None
    assert handles[0].path.read_bytes() == image_bytes

    helper = SceneShaderHelper(device)
    module, _ = helper.prepare(scene, "falcor2.tools.materials.scene_sampling_tools")
    collect = module["material_collect_properties_simple"].as_func().write(helper.bind_scene)
    result = collect(
        material,
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
    )
    cursor = result.cursor()
    cursor.load()
    color = cursor[0].read()["material_color"]
    assert color.z > color.x
    assert color.z > color.y
