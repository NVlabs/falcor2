// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gltf_material.h"
#include "material_texture_utils.h"
#include "shared_gltf_texture_selectors.h"
#include "gltf_ggx_energy.h"
#include "gltf_sheen_energy.h"

#include "falcor2/core/properties.h"
#include "falcor2/render/scene.h"
#include "falcor2/render/scene_globals.h"
#include <algorithm>
#include <bit>
#include <cmath>

namespace falcor {

namespace {

namespace F = shared::gltfmat::FeatureBits;
namespace T = shared::gltfmat::TextureBits;

template<typename CursorT>
void write_texture_info(CursorT cursor, const GLTFTextureBinding& binding, bool normal_map = false)
{
    cursor["handle"] = binding.handle;
    cursor["packed"] = binding.effective_texcoord
        | (normal_map && is_normal_map_encoded(binding.handle) ? shared::gltfmat::TextureInfoBits::NORMAL_MAP_ENCODED
                                                               : 0u);
}

} // namespace

/// Immutable scattering lookups shared by the scene's glTF materials.
class GLTFScatteringGlobals : public SceneGlobals {
    FALCOR_OBJECT(GLTFScatteringGlobals)
public:
    GLTFScatteringGlobals(sgl::Device* device)
    {
        const uint32_t size = detail::GLTF_GGX_ENERGY_SIZE;
        sgl::SubresourceData data = {
            .data = detail::GLTF_GGX_ENERGY,
            .size = sizeof(detail::GLTF_GGX_ENERGY),
            .row_pitch = size * 2 * sizeof(float),
            .slice_pitch = size * size * 2 * sizeof(float),
        };
        m_texture = device->create_texture({
            .type = sgl::TextureType::texture_2d,
            .format = sgl::Format::rg32_float,
            .width = size,
            .height = size,
            .usage = sgl::TextureUsage::shader_resource,
            .label = "GLTF GGX energy",
            .data = std::span{&data, 1},
        });
        sgl::SubresourceData sheen_data = {
            .data = detail::GLTF_SHEEN_ENERGY,
            .size = sizeof(detail::GLTF_SHEEN_ENERGY),
            .row_pitch = size * sizeof(float),
            .slice_pitch = size * size * sizeof(float),
        };
        m_sheen_texture = device->create_texture({
            .type = sgl::TextureType::texture_2d,
            .format = sgl::Format::r32_float,
            .width = size,
            .height = size,
            .usage = sgl::TextureUsage::shader_resource,
            .label = "GLTF Charlie energy",
            .data = std::span{&sheen_data, 1},
        });
        m_sampler = device->create_sampler({
            .min_filter = sgl::TextureFilteringMode::linear,
            .mag_filter = sgl::TextureFilteringMode::linear,
            .address_u = sgl::TextureAddressingMode::clamp_to_edge,
            .address_v = sgl::TextureAddressingMode::clamp_to_edge,
        });
    }

    void bind(sgl::ShaderCursor globals) const override
    {
        auto cursor = globals.find_field("gltf_scattering_globals");
        if (!cursor.is_valid())
            return;
        cursor["sheen_energy"] = m_sheen_texture->descriptor_handle_ro();
        cursor["energy"] = m_texture->descriptor_handle_ro();
        cursor["sampler"] = m_sampler->descriptor_handle();
    }

private:
    ref<sgl::Texture> m_texture;
    ref<sgl::Texture> m_sheen_texture;
    ref<sgl::Sampler> m_sampler;
};

GLTFMaterial::GLTFMaterial()
{
    set_compile_masks({F::ALL_MR, T::ALL, T::ALL, T::ALL});
}

GLTFMaterial::~GLTFMaterial() { }

void GLTFMaterial::set_compile_masks(EffectiveMasks masks)
{
    set_slang_type_name(
        "GLTFMaterial<" + std::to_string(masks.features) + "," + std::to_string(masks.textures) + ","
        + std::to_string(masks.transforms) + "," + std::to_string(masks.nonzero_texcoords) + ">"
    );
}

void GLTFMaterial::on_load_resources()
{
    // Texture load formats follow the fixed mask order in texture_bindings().
    static constexpr std::array flags{
        TextureLoadFlags::color,    // base color
        TextureLoadFlags::data_rgb, // metallic roughness
        TextureLoadFlags::normal,
        TextureLoadFlags::color,    // emissive
        TextureLoadFlags::data,     // specular
        TextureLoadFlags::color,    // specular color
        TextureLoadFlags::data,     // transmission
        TextureLoadFlags::data,     // clearcoat
        TextureLoadFlags::data_rgb, // clearcoat roughness
        TextureLoadFlags::normal,   // clearcoat normal
        TextureLoadFlags::color,    // sheen color
        TextureLoadFlags::data,     // sheen roughness
        TextureLoadFlags::data,     // iridescence
        TextureLoadFlags::data_rgb, // iridescence thickness
        TextureLoadFlags::data_rgb, // anisotropy
        TextureLoadFlags::data,     // diffuse transmission
        TextureLoadFlags::color,    // diffuse transmission color
    };
    auto bindings = texture_bindings(*this);
    static_assert(flags.size() == bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i)
        bindings[i]->load(m_scene->texture_manager(), flags[i]);
}

void GLTFMaterial::update(SceneUpdateContext& ctx)
{
    if (!m_scattering_globals)
        m_scattering_globals = m_scene->get_or_create_scene_globals(
            "GLTFScatteringGlobals",
            [device = ctx.device()]() -> ref<SceneGlobals>
            {
                return make_ref<GLTFScatteringGlobals>(device);
            }
        );

    // Derive work from resolved resources and authored factors each time the
    // material changes. In particular, factor-only edits can move tail records.
    m_feature_mask = 0;
    if (m_core.normal.handle.is_valid())
        m_feature_mask |= F::NORMAL;
    if (m_core.emissive_factor.x > 0.f || m_core.emissive_factor.y > 0.f || m_core.emissive_factor.z > 0.f)
        m_feature_mask |= F::EMISSION;
    if (m_core.use_vertex_color)
        m_feature_mask |= F::VERTEX_COLOR;
    if (float(m_core.metallic_factor) > 0.f)
        m_feature_mask |= F::MULTISCATTER;
    if (float(m_specular.factor) != 1.f || float(m_specular.color_factor.x) != 1.f
        || float(m_specular.color_factor.y) != 1.f || float(m_specular.color_factor.z) != 1.f
        || m_specular.texture.handle.is_valid() || m_specular.color_texture.handle.is_valid())
        m_feature_mask |= F::SPECULAR;
    if (float(m_transmission.factor) > 0.f)
        m_feature_mask |= F::TRANSMISSION;
    if (float(m_core.ior) != 1.5f)
        m_feature_mask |= F::IOR;
    if (m_volume.thickness_factor > 0.f)
        m_feature_mask |= F::VOLUME;
    if (float(m_clearcoat.factor) > 0.f)
        m_feature_mask |= F::CLEARCOAT;
    if (m_sheen.color_factor.x > 0.f || m_sheen.color_factor.y > 0.f || m_sheen.color_factor.z > 0.f)
        m_feature_mask |= F::SHEEN;
    if (m_iridescence.factor > 0.f)
        m_feature_mask |= F::IRIDESCENCE;
    if (m_anisotropy.strength > 0.f)
        m_feature_mask |= F::ANISOTROPY;
    if (m_diffuse_transmission.factor > 0.f)
        m_feature_mask |= F::DIFFUSE_TRANSMISSION;

    m_feature_mask &= m_feature_mask_filter & static_cast<uint32_t>(m_scene->config().gltf_allowed_features);
    if (m_feature_mask & F::VOLUME)
        FALCOR_CHECK(m_volume.attenuation_distance > 0.f, "attenuation_distance must be positive");

    static constexpr std::array<uint32_t, 17> feature_for_texture{
        0u,
        0u,
        F::NORMAL,
        F::EMISSION,
        F::SPECULAR,
        F::SPECULAR,
        F::TRANSMISSION,
        F::CLEARCOAT,
        F::CLEARCOAT,
        F::CLEARCOAT,
        F::SHEEN,
        F::SHEEN,
        F::IRIDESCENCE,
        F::IRIDESCENCE,
        F::ANISOTROPY,
        F::DIFFUSE_TRANSMISSION,
        F::DIFFUSE_TRANSMISSION,
    };
    const auto bindings = texture_bindings(*this);
    static_assert(feature_for_texture.size() == bindings.size());
    m_texture_mask = 0;
    m_transform_mask = 0;
    m_nonzero_texcoord_mask = 0;
    for (size_t index = 0; index < bindings.size(); ++index) {
        const GLTFTextureBinding& binding = *bindings[index];
        if (!binding.handle.is_valid()
            || (feature_for_texture[index] != 0 && (m_feature_mask & feature_for_texture[index]) == 0))
            continue;
        const uint32_t bit = 1u << index;
        m_texture_mask |= bit;
        if (!binding.transform.is_identity())
            m_transform_mask |= bit;
        if (binding.effective_texcoord != 0)
            m_nonzero_texcoord_mask |= bit;
    }

    const uint32_t extension_mask = m_feature_mask & F::EXTENSION_BITS;
    constexpr uint32_t feature_record_size = 32;
    constexpr uint32_t transform_record_size = 24;
    const size_t feature_bytes = std::popcount(extension_mask) * feature_record_size;
    const size_t transform_bytes = std::popcount(m_transform_mask) * transform_record_size;
    const size_t used_bytes = feature_bytes + transform_bytes;
    ensure_data_size(used_bytes);
    if (used_bytes == 0)
        return;

    // Always rewrite the active region, including the padding in fixed slots.
    // Retained allocation capacity is also zeroed so a shrunken tail cannot
    // expose stale feature data when another edit makes it active again.
    auto bytes = mutable_data();
    std::fill(bytes.begin(), bytes.end(), uint8_t(0));
    auto layout = m_scene->render_module()->layout();
    size_t record_offset = 0;
    auto write_record = [&](uint32_t bit, const char* type_name, auto&& writer)
    {
        if ((extension_mask & bit) == 0)
            return;
        const std::string structured_type = std::string("StructuredBuffer<") + type_name + ">";
        auto type = layout->find_type_by_name(structured_type.c_str());
        FALCOR_CHECK(type, "Failed to find {} type", type_name);
        auto element = layout->get_type_layout(type)->element_type_layout();
        FALCOR_CHECK_EQ(element->stride(), feature_record_size);
        auto cursor = make_ref<sgl::BufferCursor>(
            ctx.device()->type(),
            element,
            bytes.data() + record_offset,
            feature_record_size
        );
        writer((*cursor)[0]);
        cursor->apply();
        record_offset += feature_record_size;
    };
    write_record(
        F::SPECULAR,
        "GLTFSpecularRecord",
        [&](auto record)
        {
            write_texture_info(record["specular"], m_specular.texture);
            write_texture_info(record["specular_color"], m_specular.color_texture);
            record["color_strength"] = float4(float3(m_specular.color_factor), float(m_specular.factor));
        }
    );
    write_record(
        F::TRANSMISSION,
        "GLTFTransmissionRecord",
        [&](auto record)
        {
            write_texture_info(record["texture"], m_transmission.texture);
            record["factor_and_reserved"] = float4(float(m_transmission.factor), 0.f, 0.f, 0.f);
        }
    );
    write_record(
        F::IOR,
        "GLTFIORRecord",
        [&](auto record)
        {
            record["ior_and_reserved"] = float4(float(m_core.ior), 0.f, 0.f, 0.f);
        }
    );
    write_record(
        F::VOLUME,
        "GLTFVolumeRecord",
        [&](auto record)
        {
            float3 sigma_a{0.f};
            if (std::isfinite(m_volume.attenuation_distance)) {
                for (int channel = 0; channel < 3; ++channel) {
                    const float color = std::clamp(m_volume.attenuation_color[channel], 0.f, 1.f);
                    sigma_a[channel] = -std::log(color) / m_volume.attenuation_distance;
                }
            }
            record["sigma_a"] = float4(sigma_a, 0.f);
        }
    );
    write_record(
        F::CLEARCOAT,
        "GLTFClearcoatRecord",
        [&](auto record)
        {
            write_texture_info(record["texture"], m_clearcoat.texture);
            write_texture_info(record["roughness_texture"], m_clearcoat.roughness_texture);
            write_texture_info(record["normal_texture"], m_clearcoat.normal_texture, true);
            record["factors"] = float16_t4(
                m_clearcoat.factor,
                m_clearcoat.roughness_factor,
                m_clearcoat.normal_texture_scale,
                float16_t(0.f)
            );
        }
    );
    write_record(
        F::SHEEN,
        "GLTFSheenRecord",
        [&](auto record)
        {
            write_texture_info(record["color_texture"], m_sheen.color_texture);
            write_texture_info(record["roughness_texture"], m_sheen.roughness_texture);
            record["color_roughness"] = float4(m_sheen.color_factor, m_sheen.roughness_factor);
        }
    );
    write_record(
        F::IRIDESCENCE,
        "GLTFIridescenceRecord",
        [&](auto record)
        {
            write_texture_info(record["texture"], m_iridescence.texture);
            write_texture_info(record["thickness_texture"], m_iridescence.thickness_texture);
            record["factor_ior_thickness"] = float4(
                m_iridescence.factor,
                m_iridescence.ior,
                m_iridescence.thickness_minimum,
                m_iridescence.thickness_maximum
            );
        }
    );
    write_record(
        F::ANISOTROPY,
        "GLTFAnisotropyRecord",
        [&](auto record)
        {
            write_texture_info(record["texture"], m_anisotropy.texture);
            record["strength_rotation"] = float4(m_anisotropy.strength, m_anisotropy.rotation, 0.f, 0.f);
        }
    );
    write_record(
        F::DIFFUSE_TRANSMISSION,
        "GLTFDiffuseTransmissionRecord",
        [&](auto record)
        {
            write_texture_info(record["texture"], m_diffuse_transmission.texture);
            write_texture_info(record["color_texture"], m_diffuse_transmission.color_texture);
            record["color_factor"] = float4(m_diffuse_transmission.color_factor, m_diffuse_transmission.factor);
        }
    );
    FALCOR_CHECK_EQ(record_offset, feature_bytes);

    if (transform_bytes != 0) {
        auto type = layout->find_type_by_name("StructuredBuffer<GLTFTextureUVTransform>");
        FALCOR_CHECK(type, "Failed to find GLTFTextureUVTransform type");
        auto element = layout->get_type_layout(type)->element_type_layout();
        FALCOR_CHECK_EQ(element->stride(), transform_record_size);
        auto cursor
            = make_ref<sgl::BufferCursor>(ctx.device()->type(), element, bytes.data() + feature_bytes, transform_bytes);
        write_gltf_texture_binding_transforms(
            *cursor,
            bindings.size(),
            [&bindings](size_t index) -> const GLTFTextureBinding&
            {
                return *bindings[index];
            },
            m_transform_mask,
            m_scene->config().uv_origin
        );
    }
}
shared::MaterialFlags GLTFMaterial::flags() const
{
    shared::MaterialFlags result = shared::MaterialFlags::none;
    if (m_core.double_sided || (m_feature_mask & F::VOLUME) != 0)
        result |= shared::MaterialFlags::two_sided;
    if ((m_feature_mask & F::VOLUME) == 0)
        result |= shared::MaterialFlags::thin_walled;
    return result;
}

Material::OpacityDesc GLTFMaterial::opacity_desc() const
{
    shared::OpacityFlags opacity_flags = shared::OpacityFlags::none;
    if (m_core.alpha_mode == AlphaMode::mask)
        opacity_flags = shared::OpacityFlags::enabled | shared::OpacityFlags::use_threshold;
    else if (m_core.alpha_mode == AlphaMode::blend)
        opacity_flags = shared::OpacityFlags::enabled;
    if (opacity_flags != shared::OpacityFlags::none
        && (((m_feature_mask & F::VERTEX_COLOR) != 0)
            || (m_core.base_color.handle.is_valid()
                && (m_core.base_color.effective_texcoord != 0 || !m_core.base_color.transform.is_identity()))))
        opacity_flags |= shared::OpacityFlags::evaluate_material;

    return {
        .flags = opacity_flags,
        .texture_handle = m_core.base_color.handle,
        .texture_channel = 3,
        .factor = m_core.alpha_factor,
        .threshold = m_core.alpha_cutoff,
    };
}

SceneCapabilities GLTFMaterial::required_capabilities() const
{
    const bool uses_geometry_properties = (m_feature_mask & F::VERTEX_COLOR) != 0 || m_nonzero_texcoord_mask != 0;
    const bool requires_solid_transmission
        = (m_feature_mask & F::VOLUME) != 0 && (m_feature_mask & (F::TRANSMISSION | F::DIFFUSE_TRANSMISSION)) != 0;
    return material_capabilities(uses_geometry_properties, requires_solid_transmission);
}

template<typename CursorT>
void GLTFMaterial::write_to_cursor_impl(CursorT cursor) const
{
    cursor["base_color_handle"] = m_core.base_color.handle;
    cursor["base_color_packed"] = m_core.base_color.effective_texcoord;
    cursor["metallic_roughness_handle"] = m_core.metallic_roughness.handle;
    cursor["metallic_roughness_packed"] = m_core.metallic_roughness.effective_texcoord;
    cursor["normal_handle"] = (m_feature_mask & F::NORMAL) != 0 ? m_core.normal.handle : TextureHandle{};
    cursor["normal_packed"] = m_core.normal.effective_texcoord
        | (is_normal_map_encoded(m_core.normal.handle) ? shared::gltfmat::TextureInfoBits::NORMAL_MAP_ENCODED : 0u);
    cursor["emissive_handle"] = (m_feature_mask & F::EMISSION) != 0 ? m_core.emissive.handle : TextureHandle{};
    cursor["emissive_packed"] = m_core.emissive.effective_texcoord;
    cursor["base_color_factor"] = float16_t4(m_core.base_color_factor, 0.f);
    cursor["core_factors"]
        = float16_t4(m_core.metallic_factor, m_core.roughness_factor, m_core.normal_texture_scale, float16_t(0.f));
    cursor["emissive_factor"] = float4(m_core.emissive_factor, 0.f);
    cursor["feature_mask"] = m_feature_mask;
    cursor["transform_mask"] = m_transform_mask;
    cursor["extra_data"] = m_scene ? data_ref() : BufferPtr{};
}

template void GLTFMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void GLTFMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_MATERIAL(GLTFMaterial);

} // namespace falcor
