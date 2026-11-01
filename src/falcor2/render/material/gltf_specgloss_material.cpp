// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gltf_specgloss_material.h"
#include "material_texture_utils.h"
#include "shared_gltf_texture_selectors.h"

#include "falcor2/core/properties.h"
#include "falcor2/render/scene.h"
#include <algorithm>
#include <bit>
#include <utility>

namespace falcor {

namespace {

namespace F = shared::gltfmat::FeatureBits;
namespace T = shared::gltfmat::SpecGlossTextureBits;

} // namespace

GLTFSpecGlossMaterial::GLTFSpecGlossMaterial()
{
    set_compile_masks({F::ALL_SPEC_GLOSS, T::ALL, T::ALL, T::ALL});
}

GLTFSpecGlossMaterial::~GLTFSpecGlossMaterial() { }

void GLTFSpecGlossMaterial::set_compile_masks(EffectiveMasks masks)
{
    set_slang_type_name(
        "GLTFSpecGlossMaterial<" + std::to_string(masks.features) + "," + std::to_string(masks.textures) + ","
        + std::to_string(masks.transforms) + "," + std::to_string(masks.nonzero_texcoords) + ">"
    );
}

void GLTFSpecGlossMaterial::on_load_resources()
{
    // Preserve the resource-registration order while keeping UV transform slots
    // in the shader's diffuse/specular-glossiness/normal/emissive order.
    static constexpr std::array loads{
        std::pair{TextureSlot::diffuse, TextureLoadFlags::color},
        std::pair{TextureSlot::normal, TextureLoadFlags::normal},
        std::pair{TextureSlot::specular_glossiness, TextureLoadFlags::color},
        std::pair{TextureSlot::emissive, TextureLoadFlags::color},
    };
    for (const auto& [slot, flags] : loads)
        texture_binding(slot).load(m_scene->texture_manager(), flags);
}

void GLTFSpecGlossMaterial::update(SceneUpdateContext& ctx)
{
    // Every authored edit can change the exact key or the effective tail.
    m_feature_mask = 0;
    if (texture_binding(TextureSlot::normal).handle.is_valid())
        m_feature_mask |= F::NORMAL;
    if (m_core.emissive_factor.x > 0.f || m_core.emissive_factor.y > 0.f || m_core.emissive_factor.z > 0.f)
        m_feature_mask |= F::EMISSION;
    if (m_core.use_vertex_color)
        m_feature_mask |= F::VERTEX_COLOR;
    m_feature_mask &= m_feature_mask_filter & static_cast<uint32_t>(m_scene->config().gltf_allowed_features);

    static constexpr std::array<uint32_t, 4> required_feature{0u, 0u, F::NORMAL, F::EMISSION};
    m_texture_mask = 0;
    m_transform_mask = 0;
    m_nonzero_texcoord_mask = 0;
    for (size_t index = 0; index < m_textures.size(); ++index) {
        const GLTFTextureBinding& texture = m_textures[index];
        if (!texture.handle.is_valid()
            || (required_feature[index] != 0 && (m_feature_mask & required_feature[index]) == 0))
            continue;
        const uint32_t bit = 1u << index;
        m_texture_mask |= bit;
        if (!texture.transform.is_identity())
            m_transform_mask |= bit;
        if (texture.effective_texcoord != 0)
            m_nonzero_texcoord_mask |= bit;
    }

    constexpr uint32_t transform_record_size = 24;
    const size_t byte_size = std::popcount(m_transform_mask) * transform_record_size;
    ensure_data_size(byte_size);
    if (byte_size == 0)
        return;

    const auto binding_at = [this](size_t index) -> const GLTFTextureBinding&
    {
        return m_textures[index];
    };
    auto layout = m_scene->render_module()->layout();
    auto type = layout->find_type_by_name("StructuredBuffer<GLTFTextureUVTransform>");
    FALCOR_CHECK(type, "Failed to find GLTFTextureUVTransform type");
    auto element_layout = layout->get_type_layout(type)->element_type_layout();
    FALCOR_CHECK_EQ(element_layout->stride(), transform_record_size);
    auto bytes = mutable_data();
    std::fill(bytes.begin(), bytes.end(), uint8_t(0));
    auto cursor = make_ref<sgl::BufferCursor>(ctx.device()->type(), element_layout, bytes.data(), byte_size);
    write_gltf_texture_binding_transforms(
        *cursor,
        m_textures.size(),
        binding_at,
        m_transform_mask,
        m_scene->config().uv_origin
    );
}

shared::MaterialFlags GLTFSpecGlossMaterial::flags() const
{
    shared::MaterialFlags result = shared::MaterialFlags::none;
    if (m_core.double_sided)
        result |= shared::MaterialFlags::two_sided;
    return result;
}

Material::OpacityDesc GLTFSpecGlossMaterial::opacity_desc() const
{
    shared::OpacityFlags opacity_flags = shared::OpacityFlags::none;
    if (m_core.alpha_mode == AlphaMode::mask)
        opacity_flags = shared::OpacityFlags::enabled | shared::OpacityFlags::use_threshold;
    else if (m_core.alpha_mode == AlphaMode::blend)
        opacity_flags = shared::OpacityFlags::enabled;
    const auto& diffuse = texture_binding(TextureSlot::diffuse);
    if (opacity_flags != shared::OpacityFlags::none
        && ((m_feature_mask & F::VERTEX_COLOR) != 0
            || (diffuse.handle.is_valid() && (diffuse.effective_texcoord != 0 || !diffuse.transform.is_identity()))))
        opacity_flags |= shared::OpacityFlags::evaluate_material;

    return {
        .flags = opacity_flags,
        .texture_handle = texture_binding(TextureSlot::diffuse).handle,
        .texture_channel = 3,
        .factor = m_core.alpha_factor,
        .threshold = m_core.alpha_cutoff,
    };
}

SceneCapabilities GLTFSpecGlossMaterial::required_capabilities() const
{
    return material_capabilities((m_feature_mask & F::VERTEX_COLOR) != 0 || m_nonzero_texcoord_mask != 0, false);
}

template<typename CursorT>
void GLTFSpecGlossMaterial::write_to_cursor_impl(CursorT cursor) const
{
    const auto& diffuse = texture_binding(TextureSlot::diffuse);
    const auto& specular = texture_binding(TextureSlot::specular_glossiness);
    const auto& normal = texture_binding(TextureSlot::normal);
    const auto& emissive = texture_binding(TextureSlot::emissive);
    cursor["diffuse_handle"] = diffuse.handle;
    cursor["diffuse_packed"] = diffuse.effective_texcoord;
    cursor["specular_glossiness_handle"] = specular.handle;
    cursor["specular_glossiness_packed"] = specular.effective_texcoord;
    cursor["normal_handle"] = (m_feature_mask & F::NORMAL) != 0 ? normal.handle : TextureHandle{};
    cursor["normal_packed"] = normal.effective_texcoord
        | (is_normal_map_encoded(normal.handle) ? shared::gltfmat::TextureInfoBits::NORMAL_MAP_ENCODED : 0u);
    cursor["emissive_handle"] = (m_feature_mask & F::EMISSION) != 0 ? emissive.handle : TextureHandle{};
    cursor["emissive_packed"] = emissive.effective_texcoord;
    cursor["diffuse_factor"] = float16_t4(m_core.diffuse_factor, 0.f);
    cursor["specular_factor"] = float16_t4(m_core.specular_factor, 0.f);
    cursor["emissive_factor"] = float4(m_core.emissive_factor, 0.f);
    cursor["core_factors"]
        = float16_t4(m_core.glossiness_factor, m_core.normal_texture_scale, float16_t(0.f), float16_t(0.f));
    cursor["feature_mask"] = m_feature_mask;
    cursor["transform_mask"] = m_transform_mask;
    cursor["extra_data"] = m_scene ? data_ref() : BufferOffsetPair{};
}

template void GLTFSpecGlossMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void GLTFSpecGlossMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_MATERIAL(GLTFSpecGlossMaterial);

} // namespace falcor
