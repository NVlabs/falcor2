// SPDX-License-Identifier: Apache-2.0

#include "standard_specgloss_material.h"

#include "falcor2/core/properties.h"
#include "falcor2/render/scene.h"

namespace falcor {

StandardSpecGlossMaterial::StandardSpecGlossMaterial()
{
    set_slang_type_name("StandardSpecGlossMaterial");
}

StandardSpecGlossMaterial::~StandardSpecGlossMaterial() { }

void StandardSpecGlossMaterial::on_load_resources()
{
    if (m_diffuse_texture) {
        m_diffuse_texture_handle = m_scene->texture_manager()->register_texture({
            .texture = m_diffuse_texture,
        });
    } else if (!m_diffuse_texture_path.empty()) {
        m_diffuse_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_diffuse_texture_path,
            .load_deferred = true,
        });
    } else {
        m_diffuse_texture_handle = {};
    }
    if (m_specular_glossiness_texture) {
        m_specular_glossiness_texture_handle = m_scene->texture_manager()->register_texture({
            .texture = m_specular_glossiness_texture,
        });
    } else if (!m_specular_glossiness_texture_path.empty()) {
        m_specular_glossiness_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_specular_glossiness_texture_path,
            .load_deferred = true,
        });
    } else {
        m_specular_glossiness_texture_handle = {};
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

void StandardSpecGlossMaterial::update(SceneUpdateContext& ctx)
{
    FALCOR_UNUSED(ctx);
}

template<typename CursorT>
void StandardSpecGlossMaterial::write_to_cursor_impl(CursorT cursor) const
{
    cursor["diffuse_texture"] = m_diffuse_texture_handle;
    cursor["specular_glossiness_texture"] = m_specular_glossiness_texture_handle;
    cursor["normal_texture"] = m_normal_texture_handle;
    cursor["emissive_texture"] = m_emissive_texture_handle;
    cursor["transmission_texture"] = m_transmission_texture_handle;
    cursor["diffuse_factor"] = m_diffuse_factor;
    cursor["specular_factor"] = m_specular_factor;
    cursor["glossiness_factor"] = m_glossiness_factor;
    cursor["normal_texture_scale"] = m_normal_texture_scale;
    cursor["emissive_factor"] = m_emissive_factor;
    cursor["ior"] = m_ior;
    cursor["transmission_factor"] = m_transmission_factor;
    cursor["diffuse_transmission_factor"] = m_diffuse_transmission_factor;
    cursor["specular_transmission_factor"] = m_specular_transmission_factor;
    cursor["thin_surface"] = m_thin_surface;
    cursor["double_sided"] = m_double_sided;
}

template void StandardSpecGlossMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void StandardSpecGlossMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_MATERIAL(StandardSpecGlossMaterial);

} // namespace falcor
