# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper


_STATE_SOURCE = """\
import falcor2.render;

public uint3 specgloss_state(uint material_id, uint feature_word, uint transform_word, uint extra_offset_word)
{
    const MaterialData material = g_scene.material_system.get_material_data(MaterialID(material_id));
    return uint3(
        material.payload.data[feature_word],
        material.payload.data[transform_word],
        material.payload.data[extra_offset_word]
    );
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specgloss_exact_type_tracks_live_texture_uv_edits(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.GLTFSpecGlossMaterial)
    source = spy.Module(device.load_module_from_source("specgloss_live_state_probe", _STATE_SOURCE))
    helper = SceneShaderHelper(device)

    def check(textures: int, transforms: int, nonzero_texcoords: int) -> None:
        scene.update()
        expected = f"GLTFSpecGlossMaterial<0,{textures},{transforms},{nonzero_texcoords}>"
        assert material.slang_type_name == expected
        assert any(
            item.interface_name == "IMaterial" and item.type_name == expected
            for item in scene.requirements.type_conformances
        )
        module, _ = helper.prepare(scene, source)
        layout = module.layout.program_layout
        buffer_type = layout.find_type_by_name(f"StructuredBuffer<{expected}>")
        assert buffer_type is not None
        material_layout = layout.get_type_layout(buffer_type).element_type_layout
        assert material_layout.stride <= 128

        def payload_word(*path: str) -> int:
            field_layout = material_layout
            offset = 0
            for name in path:
                field = next(field for field in field_layout.fields if field.name == name)
                offset += field.offset
                field_layout = field.type_layout
            assert offset >= 16 and offset % 4 == 0
            return offset // 4 - 4

        words = (
            payload_word("feature_mask"),
            payload_word("transform_mask"),
            payload_word("extra_data", "offset"),
        )
        uploaded = module.specgloss_state.as_func().write(helper.bind_scene)(
            int(material.material_id), *words
        )
        assert int(uploaded.x) == 0  # No normal, emission, or vertex color.
        assert int(uploaded.y) == transforms
        if transforms == 0:
            assert int(uploaded.z) == 0  # No transform tail allocation.

    check(0, 0, 0)
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=1,
        height=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.ones((1, 1, 4), dtype=np.float32),
    )
    # Authored transforms have no GPU tail until the texture is present.
    material.diffuse_texture_offset = spy.float2(0.125, 0.25)
    check(0, 0, 0)
    material.diffuse_texture = texture
    check(1, 1, 0)

    material.diffuse_texture_offset = spy.float2(0)
    check(1, 0, 0)
    material.diffuse_texture_texcoord = 1
    check(1, 0, 1)

    material.diffuse_texture_offset = spy.float2(0.125, 0.25)
    check(1, 1, 1)

    material.diffuse_texture_offset = spy.float2(0)
    check(1, 0, 1)
