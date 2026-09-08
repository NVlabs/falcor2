# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import falcor2 as f2
import falcor2.testing.helpers as helpers
import pytest
import slangpy as spy


_LARGE_EDITABLE_DOCUMENT = """\
<materialx version="1.39">
  <gltf_pbr name="shader" type="surfaceshader" nodedef="ND_gltf_pbr_surfaceshader">
    <input name="base_color" type="color3" nodename="base_color_image" output="outcolor" />
    <input name="metallic" type="float" nodename="metallic_extract" />
    <input name="roughness" type="float" nodename="roughness_extract" />
    <input name="occlusion" type="float" nodename="occlusion_extract" />
    <input name="transmission" type="float" value="0" />
    <input name="specular" type="float" value="1" />
    <input name="specular_color" type="color3" value="1, 1, 1" />
    <input name="ior" type="float" value="1.5" />
    <input name="alpha" type="float" value="1" />
    <input name="alpha_mode" type="integer" value="0" />
    <input name="alpha_cutoff" type="float" value="0.5" />
    <input name="iridescence" type="float" value="0" />
    <input name="iridescence_ior" type="float" value="1.3" />
    <input name="iridescence_thickness" type="float" value="100" />
    <input name="sheen_color" type="color3" value="0, 0, 0" />
    <input name="sheen_roughness" type="float" value="0" />
    <input name="clearcoat" type="float" value="0" />
    <input name="clearcoat_roughness" type="float" value="0" />
    <input name="emissive" type="color3" nodename="emissive_image" output="outcolor" />
    <input name="emissive_strength" type="float" value="1" />
    <input name="thickness" type="float" value="0" />
    <input name="attenuation_color" type="color3" value="1, 1, 1" />
    <input name="anisotropy_strength" type="float" value="0" />
    <input name="anisotropy_rotation" type="float" value="0" />
    <input name="dispersion" type="float" value="0" />
    <input name="normal" type="vector3" nodename="normal_image" />
  </gltf_pbr>
  <gltf_colorimage name="base_color_image" type="multioutput">
    <input name="file" type="filename" value="" colorspace="srgb_texture" />
    <output name="outcolor" type="color3" />
    <output name="outa" type="float" />
  </gltf_colorimage>
  <gltf_colorimage name="emissive_image" type="multioutput">
    <input name="file" type="filename" value="" colorspace="srgb_texture" />
    <output name="outcolor" type="color3" />
    <output name="outa" type="float" />
  </gltf_colorimage>
  <gltf_image name="orm_image" type="vector3">
    <input name="file" type="filename" value="" />
  </gltf_image>
  <extract name="metallic_extract" type="float">
    <input name="in" type="vector3" nodename="orm_image" />
    <input name="index" type="integer" value="2" />
  </extract>
  <extract name="roughness_extract" type="float">
    <input name="in" type="vector3" nodename="orm_image" />
    <input name="index" type="integer" value="1" />
  </extract>
  <extract name="occlusion_extract" type="float">
    <input name="in" type="vector3" nodename="orm_image" />
    <input name="index" type="integer" value="0" />
  </extract>
  <gltf_normalmap name="normal_image" type="vector3">
    <input name="file" type="filename" value="" />
  </gltf_normalmap>
  <surfacematerial name="material" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="shader" />
  </surfacematerial>
</materialx>
"""


def _macro_value(shader: str, name: str) -> int:
    match = re.search(rf"^\s*#define {re.escape(name)}\s+(\d+)\s*$", shader, flags=re.MULTILINE)
    assert match is not None
    return int(match.group(1))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_materialx_data_size_bound_and_buffer_write(
    device_type: spy.DeviceType, tmp_path: Path
) -> None:
    shader_path = tmp_path / f"material_data_size_{device_type.name}.slang"
    device = helpers.get_device(device_type, use_cache=False)
    try:
        scene = f2.Scene.create(device)
        material = scene.create_material(
            f2.MaterialXMaterial,
            f2.Properties(
                {
                    "mtlx_buffer": _LARGE_EDITABLE_DOCUMENT,
                    "mtlx_node_name": "material",
                    "mtlx_editable_params": "*",
                    "debug_write_shader_path": str(shader_path),
                }
            ),
        )

        # This updates the out-of-line buffer through a reflected BufferCursor.
        scene.update()

        assert material.slang_type_name
        shader = shader_path.read_text(encoding="utf-8")
        estimated_size = _macro_value(shader, "MATERIALX_PAYLOAD_SIZE")
        payload_limit = _macro_value(shader, "MATERIAL_PAYLOAD_LIMIT")
        assert estimated_size > payload_limit

        module_name = next(
            line.removeprefix("module ").removesuffix(";")
            for line in shader.splitlines()
            if line.startswith("module ")
        )
        data_name = next(
            line.split()[2]
            for line in shader.splitlines()
            if line.startswith("    public struct ") and "_Data" in line
        )
        module = device.load_module_from_source(module_name, shader)
        buffer_type = module.layout.find_type_by_name(f"StructuredBuffer<mtlx::{data_name}>")
        assert buffer_type is not None
        reflected_stride = module.layout.get_type_layout(buffer_type).element_type_layout.stride

        assert estimated_size >= reflected_stride
    finally:
        device.close()
