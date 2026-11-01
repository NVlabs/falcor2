// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "standard_specgloss_material.h"
#include "material_texture_utils.h"

#include "falcor2/core/properties.h"
#include "falcor2/render/scene.h"

namespace falcor {

namespace {

uint32_t pack_texture_coordinate_selectors(
    uint32_t diffuse,
    uint32_t specular_glossiness,
    uint32_t normal,
    uint32_t emissive,
    uint32_t transmission,
    bool use_vertex_color,
    bool normal_map_encoded
)
{
    return sgl::narrow_cast<uint32_t>(
        diffuse | (specular_glossiness << 3) | (normal << 6) | (emissive << 9) | (transmission << 12)
        | (uint32_t(use_vertex_color) << 15) | (uint32_t(normal_map_encoded) << 16)
    );
}

} // namespace

StandardSpecGlossMaterial::StandardSpecGlossMaterial()
{
    set_slang_type_name("StandardSpecGlossMaterial");
}

StandardSpecGlossMaterial::~StandardSpecGlossMaterial() { }

void StandardSpecGlossMaterial::on_load_resources()
{
    m_diffuse_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_diffuse_texture,
        m_diffuse_texture_sampler,
        m_diffuse_texture_path,
        TextureLoadFlags::color
    );

    m_specular_glossiness_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_specular_glossiness_texture,
        m_specular_glossiness_texture_sampler,
        m_specular_glossiness_texture_path,
        TextureLoadFlags::color
    );

    m_normal_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_normal_texture,
        m_normal_texture_sampler,
        m_normal_texture_path,
        TextureLoadFlags::normal
    );

    m_emissive_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_emissive_texture,
        m_emissive_texture_sampler,
        m_emissive_texture_path,
        TextureLoadFlags::color
    );

    m_transmission_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_transmission_texture,
        m_transmission_texture_sampler,
        m_transmission_texture_path,
        TextureLoadFlags::color
    );
}

void StandardSpecGlossMaterial::update(SceneUpdateContext& ctx)
{
    FALCOR_UNUSED(ctx);
}

shared::MaterialFlags StandardSpecGlossMaterial::flags() const
{
    shared::MaterialFlags result = shared::MaterialFlags::none;
    if (m_double_sided || m_diffuse_transmission_factor > 0.h || m_specular_transmission_factor > 0.h)
        result |= shared::MaterialFlags::two_sided;
    if (m_thin_walled)
        result |= shared::MaterialFlags::thin_walled;
    return result;
}

Material::OpacityDesc StandardSpecGlossMaterial::opacity_desc() const
{
    shared::OpacityFlags opacity_flags = shared::OpacityFlags::none;
    if (m_alpha_mode == AlphaMode::mask)
        opacity_flags = shared::OpacityFlags::enabled | shared::OpacityFlags::use_threshold;
    else if (m_alpha_mode == AlphaMode::blend)
        opacity_flags = shared::OpacityFlags::enabled;
    if (opacity_flags != shared::OpacityFlags::none && (m_diffuse_texture_texcoord != 0 || m_use_vertex_color))
        opacity_flags |= shared::OpacityFlags::evaluate_material;

    return {
        .flags = opacity_flags,
        .texture_handle = m_diffuse_texture_handle,
        .texture_channel = 3,
        .factor = m_alpha_factor,
        .threshold = m_alpha_cutoff,
    };
}

SceneCapabilities StandardSpecGlossMaterial::required_capabilities() const
{
    bool uses_geometry_properties = m_use_vertex_color || m_diffuse_texture_texcoord != 0
        || m_specular_glossiness_texture_texcoord != 0 || m_normal_texture_texcoord != 0
        || m_emissive_texture_texcoord != 0 || m_transmission_texture_texcoord != 0;
    // The specular lobe remains selectable when its tint is black; stay conservative.
    const bool has_diffuse_transmission = m_diffuse_transmission_factor > 0.h
        && (m_transmission_factor.x > 0.h || m_transmission_factor.y > 0.h || m_transmission_factor.z > 0.h);
    const bool requires_solid_transmission
        = !m_thin_walled && (m_specular_transmission_factor > 0.h || has_diffuse_transmission);
    return material_capabilities(uses_geometry_properties, requires_solid_transmission);
}

template<typename CursorT>
void StandardSpecGlossMaterial::write_to_cursor_impl(CursorT cursor) const
{
    cursor["diffuse_texture"] = m_diffuse_texture_handle;
    cursor["specular_glossiness_texture"] = m_specular_glossiness_texture_handle;
    cursor["normal_texture"] = m_normal_texture_handle;
    cursor["emissive_texture"] = m_emissive_texture_handle;
    cursor["transmission_texture"] = m_transmission_texture_handle;
    cursor["diffuse_factor"] = float16_t4(m_diffuse_factor, 0.f);
    cursor["specular_factor"] = float16_t4(m_specular_factor, 0.f);
    cursor["glossiness_factor"] = m_glossiness_factor;
    cursor["normal_texture_scale"] = m_normal_texture_scale;
    cursor["emissive_factor"] = m_emissive_factor;
    cursor["ior"] = m_ior;
    cursor["transmission_factor"] = float16_t4(m_transmission_factor, 0.f);
    cursor["diffuse_transmission_factor"] = m_diffuse_transmission_factor;
    cursor["specular_transmission_factor"] = m_specular_transmission_factor;
    cursor["volume_sigma_a"] = m_volume_sigma_a;
    cursor["volume_sigma_s"] = m_volume_sigma_s;
    cursor["volume_anisotropy"] = m_volume_anisotropy;
    cursor["texture_coordinate_selectors"] = pack_texture_coordinate_selectors(
        m_diffuse_texture_texcoord,
        m_specular_glossiness_texture_texcoord,
        m_normal_texture_texcoord,
        m_emissive_texture_texcoord,
        m_transmission_texture_texcoord,
        m_use_vertex_color,
        is_normal_map_encoded(m_normal_texture_handle)
    );
}

template void StandardSpecGlossMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void StandardSpecGlossMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_MATERIAL(StandardSpecGlossMaterial);

} // namespace falcor
