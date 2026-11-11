// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "unlit_material.h"

#include "falcor2/render/scene.h"

namespace falcor {

UnlitMaterial::UnlitMaterial()
{
    set_slang_type_name("UnlitMaterial");
}

UnlitMaterial::~UnlitMaterial() { }

void UnlitMaterial::on_load_resources()
{
    if (m_base_color_texture) {
        m_base_color_texture_handle = m_scene->texture_manager()->register_texture({
            .texture = m_base_color_texture,
            .sampler = m_base_color_texture_sampler,
        });
    } else if (!m_base_color_texture_path.empty()) {
        m_base_color_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_base_color_texture_path,
            .sampler = m_base_color_texture_sampler,
            .flags = TextureLoadFlags::color,
            .load_deferred = true,
        });
    } else {
        m_base_color_texture_handle = {};
    }
}

shared::MaterialFlags UnlitMaterial::flags() const
{
    shared::MaterialFlags result = shared::MaterialFlags::unlit;
    if (m_double_sided)
        result |= shared::MaterialFlags::two_sided;
    return result;
}

Material::OpacityDesc UnlitMaterial::opacity_desc() const
{
    shared::OpacityFlags opacity_flags = shared::OpacityFlags::none;
    if (m_alpha_mode == AlphaMode::mask)
        opacity_flags = shared::OpacityFlags::enabled | shared::OpacityFlags::use_threshold;
    else if (m_alpha_mode == AlphaMode::blend)
        opacity_flags = shared::OpacityFlags::enabled;
    if (opacity_flags != shared::OpacityFlags::none && (m_base_color_texture_texcoord != 0 || m_use_vertex_color))
        opacity_flags |= shared::OpacityFlags::evaluate_material;

    return {
        .flags = opacity_flags,
        .texture_handle = m_base_color_texture_handle,
        .texture_channel = 3,
        .factor = m_alpha_factor,
        .threshold = m_alpha_cutoff,
    };
}

SceneCapabilities UnlitMaterial::required_capabilities() const
{
    bool uses_geometry_properties = m_use_vertex_color || m_base_color_texture_texcoord != 0;
    return material_capabilities(uses_geometry_properties, false);
}

template<typename CursorT>
void UnlitMaterial::write_to_cursor_impl(CursorT cursor) const
{
    cursor["base_color_texture"] = m_base_color_texture_handle;
    cursor["base_color_texture_texcoord"] = m_base_color_texture_texcoord;
    cursor["base_color_factor"] = m_base_color_factor;
    cursor["use_vertex_color"] = m_use_vertex_color;
}

template void UnlitMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void UnlitMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_MATERIAL(UnlitMaterial);

} // namespace falcor
