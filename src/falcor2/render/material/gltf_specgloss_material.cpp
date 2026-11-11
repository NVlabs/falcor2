// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gltf_specgloss_material.h"
#include "material_texture_utils.h"
#include "shared_gltf_texture_selectors.h"

#include "falcor2/core/properties.h"
#include "falcor2/render/scene.h"
#include <bit>

namespace falcor {

namespace {

uint32_t pack_texture_coordinate_selectors(
    uint32_t diffuse,
    uint32_t specular_glossiness,
    uint32_t normal,
    uint32_t emissive,
    bool use_vertex_color,
    bool normal_map_encoded
)
{
    using Bits = shared::gltfmat::SpecGlossTCBits;
    uint32_t selectors = 0;
    Bits::diffuse::insert(selectors, diffuse);
    Bits::specular_glossiness::insert(selectors, specular_glossiness);
    Bits::normal::insert(selectors, normal);
    Bits::emissive::insert(selectors, emissive);
    Bits::use_vertex_color::insert(selectors, uint32_t(use_vertex_color));
    Bits::normal_map_encoded::insert(selectors, uint32_t(normal_map_encoded));
    return selectors;
}

} // namespace

GLTFSpecGlossMaterial::GLTFSpecGlossMaterial()
{
    set_slang_type_name("GLTFSpecGlossMaterial");
}

GLTFSpecGlossMaterial::~GLTFSpecGlossMaterial() { }

void GLTFSpecGlossMaterial::on_load_resources()
{
    m_diffuse_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_diffuse_texture,
        m_diffuse_texture_sampler,
        m_diffuse_texture_path,
        TextureLoadFlags::color
    );

    m_normal_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_normal_texture,
        m_normal_texture_sampler,
        m_normal_texture_path,
        TextureLoadFlags::normal
    );

    m_specular_glossiness_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_specular_glossiness_texture,
        m_specular_glossiness_texture_sampler,
        m_specular_glossiness_texture_path,
        TextureLoadFlags::color
    );

    m_emissive_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_emissive_texture,
        m_emissive_texture_sampler,
        m_emissive_texture_path,
        TextureLoadFlags::color
    );
}

void GLTFSpecGlossMaterial::update(SceneUpdateContext& ctx)
{
    FALCOR_UNUSED(ctx);
    if (!m_transforms_dirty)
        return;
    m_transform_mask = gltf_texture_transform_mask(m_texture_transforms);
    if (m_transform_mask == 0) {
        ensure_data_size(0);
    } else {
        auto layout = m_scene->render_module()->layout();
        auto type = layout->find_type_by_name("StructuredBuffer<GLTFTextureUVTransform>");
        FALCOR_CHECK(type, "Failed to find GLTFTextureUVTransform type");
        auto element_layout = layout->get_type_layout(type)->element_type_layout();
        FALCOR_CHECK_LE(element_layout->alignment(), 16);
        const size_t byte_size = std::popcount(m_transform_mask) * element_layout->stride();
        ensure_data_size(byte_size);
        auto cursor
            = make_ref<sgl::BufferCursor>(m_scene->device()->type(), element_layout, mutable_data().data(), byte_size);
        write_gltf_texture_transforms(*cursor, m_texture_transforms, m_scene->config().uv_origin);
    }
    m_transforms_dirty = false;
}

shared::MaterialFlags GLTFSpecGlossMaterial::flags() const
{
    shared::MaterialFlags result = shared::MaterialFlags::none;
    if (m_double_sided)
        result |= shared::MaterialFlags::two_sided;
    return result;
}

Material::OpacityDesc GLTFSpecGlossMaterial::opacity_desc() const
{
    shared::OpacityFlags opacity_flags = shared::OpacityFlags::none;
    if (m_alpha_mode == AlphaMode::mask)
        opacity_flags = shared::OpacityFlags::enabled | shared::OpacityFlags::use_threshold;
    else if (m_alpha_mode == AlphaMode::blend)
        opacity_flags = shared::OpacityFlags::enabled;
    if (opacity_flags != shared::OpacityFlags::none
        && (m_diffuse_texture_texcoord != 0 || m_use_vertex_color || !m_texture_transforms[0].is_identity()))
        opacity_flags |= shared::OpacityFlags::evaluate_material;

    return {
        .flags = opacity_flags,
        .texture_handle = m_diffuse_texture_handle,
        .texture_channel = 3,
        .factor = m_alpha_factor,
        .threshold = m_alpha_cutoff,
    };
}

SceneCapabilities GLTFSpecGlossMaterial::required_capabilities() const
{
    bool uses_geometry_properties = m_use_vertex_color || m_diffuse_texture_texcoord != 0
        || m_specular_glossiness_texture_texcoord != 0 || m_normal_texture_texcoord != 0
        || m_emissive_texture_texcoord != 0;
    return material_capabilities(uses_geometry_properties);
}

template<typename CursorT>
void GLTFSpecGlossMaterial::write_to_cursor_impl(CursorT cursor) const
{
    cursor["transform_mask"] = m_transform_mask;
    cursor["texture_transform_data"] = data_ref();
    cursor["diffuse_texture"] = m_diffuse_texture_handle;
    cursor["diffuse_factor"] = float16_t4(m_diffuse_factor, 0.f);
    cursor["specular_glossiness_texture"] = m_specular_glossiness_texture_handle;
    cursor["specular_factor"] = float16_t4(m_specular_factor, 0.f);
    cursor["glossiness_factor"] = m_glossiness_factor;
    cursor["normal_texture"] = m_normal_texture_handle;
    cursor["normal_texture_scale"] = m_normal_texture_scale;
    cursor["emissive_factor"] = m_emissive_factor;
    cursor["emissive_texture"] = m_emissive_texture_handle;
    cursor["texture_coordinate_selectors"] = pack_texture_coordinate_selectors(
        m_diffuse_texture_texcoord,
        m_specular_glossiness_texture_texcoord,
        m_normal_texture_texcoord,
        m_emissive_texture_texcoord,
        m_use_vertex_color,
        is_normal_map_encoded(m_normal_texture_handle)
    );
}

template void GLTFSpecGlossMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void GLTFSpecGlossMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_MATERIAL(GLTFSpecGlossMaterial);

} // namespace falcor
