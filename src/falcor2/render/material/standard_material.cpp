// SPDX-License-Identifier: Apache-2.0

#include "standard_material.h"

#include "falcor2/core/properties.h"
#include "falcor2/render/scene.h"

namespace falcor {

StandardMaterial::StandardMaterial()
{
    set_slang_type_name("StandardMaterial");
}

StandardMaterial::~StandardMaterial() { }

void StandardMaterial::on_load_resources()
{
    if (m_base_color_texture) {
        m_base_color_texture_handle = m_scene->texture_manager()->register_texture({
            .texture = m_base_color_texture,
        });
    } else if (!m_base_color_texture_path.empty()) {
        m_base_color_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_base_color_texture_path,
            .load_deferred = true,
        });
    } else {
        m_base_color_texture_handle = {};
    }
    if (m_normal_texture) {
        m_normal_texture_handle = m_scene->texture_manager()->register_texture({
            .texture = m_normal_texture,
        });
    } else if (!m_normal_texture_path.empty()) {
        m_normal_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_normal_texture_path,
            .srgb = false,
            .load_deferred = true,
        });
    } else {
        m_normal_texture_handle = {};
    }
    if (m_metallic_roughness_texture) {
        m_metallic_roughness_texture_handle = m_scene->texture_manager()->register_texture({
            .texture = m_metallic_roughness_texture,
        });
    } else if (!m_metallic_roughness_texture_path.empty()) {
        m_metallic_roughness_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_metallic_roughness_texture_path,
            .srgb = false,
            .load_deferred = true,
        });
    } else {
        m_metallic_roughness_texture_handle = {};
    }
    if (m_emissive_texture) {
        m_emissive_texture_handle = m_scene->texture_manager()->register_texture({
            .texture = m_emissive_texture,
        });
    } else if (!m_emissive_texture_path.empty()) {
        m_emissive_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_emissive_texture_path,
            .load_deferred = true,
        });
    } else {
        m_emissive_texture_handle = {};
    }
    if (m_transmission_texture) {
        m_transmission_texture_handle = m_scene->texture_manager()->register_texture({
            .texture = m_transmission_texture,
        });
    } else if (!m_transmission_texture_path.empty()) {
        m_transmission_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_transmission_texture_path,
            .load_deferred = true,
        });
    } else {
        m_transmission_texture_handle = {};
    }
}

void StandardMaterial::update(SceneUpdateContext& ctx)
{
    FALCOR_UNUSED(ctx);
}

template<typename CursorT>
void StandardMaterial::write_to_cursor_impl(CursorT cursor) const
{
    cursor["base_color_texture"] = m_base_color_texture_handle;
    cursor["base_color_factor"] = m_base_color_factor;
    cursor["metallic_roughness_texture"] = m_metallic_roughness_texture_handle;
    cursor["metallic_factor"] = m_metallic_factor;
    cursor["roughness_factor"] = m_roughness_factor;
    cursor["normal_texture"] = m_normal_texture_handle;
    cursor["normal_texture_scale"] = m_normal_texture_scale;
    cursor["emissive_factor"] = m_emissive_factor;
    cursor["emissive_texture"] = m_emissive_texture_handle;
    cursor["transmission_texture"] = m_transmission_texture_handle;
    cursor["ior"] = m_ior;
    cursor["transmission_factor"] = m_transmission_factor;
    cursor["diffuse_transmission_factor"] = m_diffuse_transmission_factor;
    cursor["specular_transmission_factor"] = m_specular_transmission_factor;
    cursor["thin_surface"] = m_thin_surface;
    cursor["double_sided"] = m_double_sided;
    cursor["metallic_texture_channel"] = m_metallic_texture_channel;
    cursor["roughness_texture_channel"] = m_roughness_texture_channel;
}

template void StandardMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void StandardMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_MATERIAL(StandardMaterial);

} // namespace falcor
