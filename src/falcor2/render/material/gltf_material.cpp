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
#include <bit>
#include <cmath>

namespace falcor {

namespace {

uint32_t pack_texture_coordinate_selectors(
    uint32_t base_color,
    uint32_t metallic_roughness,
    uint32_t normal,
    uint32_t emissive,
    uint32_t specular,
    uint32_t specular_color,
    uint32_t transmission,
    bool normal_map_encoded,
    bool use_vertex_color
)
{
    using Bits = shared::gltfmat::TCBits;
    uint32_t selectors = 0;
    Bits::base_color::insert(selectors, base_color);
    Bits::metallic_roughness::insert(selectors, metallic_roughness);
    Bits::normal::insert(selectors, normal);
    Bits::emissive::insert(selectors, emissive);
    Bits::specular::insert(selectors, specular);
    Bits::specular_color::insert(selectors, specular_color);
    Bits::transmission::insert(selectors, transmission);
    Bits::normal_map_encoded::insert(selectors, uint32_t(normal_map_encoded));
    Bits::use_vertex_color::insert(selectors, uint32_t(use_vertex_color));
    return selectors;
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
    set_slang_type_name("GLTFMaterial");
}

GLTFMaterial::~GLTFMaterial() { }

void GLTFMaterial::on_load_resources()
{
    m_transforms_dirty = true;
    m_anisotropy_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_anisotropy_texture,
        m_anisotropy_texture_sampler,
        m_anisotropy_texture_path,
        TextureLoadFlags::data_rgb
    );
    m_diffuse_transmission_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_diffuse_transmission_texture,
        m_diffuse_transmission_texture_sampler,
        m_diffuse_transmission_texture_path,
        TextureLoadFlags::data
    );
    m_diffuse_transmission_color_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_diffuse_transmission_color_texture,
        m_diffuse_transmission_color_texture_sampler,
        m_diffuse_transmission_color_texture_path,
        TextureLoadFlags::color
    );
    m_sheen_color_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_sheen_color_texture,
        m_sheen_color_texture_sampler,
        m_sheen_color_texture_path,
        TextureLoadFlags::color
    );
    m_sheen_roughness_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_sheen_roughness_texture,
        m_sheen_roughness_texture_sampler,
        m_sheen_roughness_texture_path,
        TextureLoadFlags::data
    );
    m_iridescence_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_iridescence_texture,
        m_iridescence_texture_sampler,
        m_iridescence_texture_path,
        TextureLoadFlags::data
    );
    m_iridescence_thickness_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_iridescence_thickness_texture,
        m_iridescence_thickness_texture_sampler,
        m_iridescence_thickness_texture_path,
        TextureLoadFlags::data_rgb
    );

    m_clearcoat_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_clearcoat_texture,
        m_clearcoat_texture_sampler,
        m_clearcoat_texture_path,
        TextureLoadFlags::data
    );
    m_clearcoat_roughness_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_clearcoat_roughness_texture,
        m_clearcoat_roughness_texture_sampler,
        m_clearcoat_roughness_texture_path,
        TextureLoadFlags::data_rgb
    );
    m_clearcoat_normal_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_clearcoat_normal_texture,
        m_clearcoat_normal_texture_sampler,
        m_clearcoat_normal_texture_path,
        TextureLoadFlags::normal
    );
    m_transmission_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_transmission_texture,
        m_transmission_texture_sampler,
        m_transmission_texture_path,
        TextureLoadFlags::data
    );
    m_specular_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_specular_texture,
        m_specular_texture_sampler,
        m_specular_texture_path,
        TextureLoadFlags::data
    );
    m_specular_color_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_specular_color_texture,
        m_specular_color_texture_sampler,
        m_specular_color_texture_path,
        TextureLoadFlags::color
    );
    m_base_color_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_base_color_texture,
        m_base_color_texture_sampler,
        m_base_color_texture_path,
        TextureLoadFlags::color
    );

    m_normal_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_normal_texture,
        m_normal_texture_sampler,
        m_normal_texture_path,
        TextureLoadFlags::normal
    );

    m_metallic_roughness_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_metallic_roughness_texture,
        m_metallic_roughness_texture_sampler,
        m_metallic_roughness_texture_path,
        TextureLoadFlags::data_rgb
    );

    m_emissive_texture_handle = load_material_texture(
        m_scene->texture_manager(),
        m_emissive_texture,
        m_emissive_texture_sampler,
        m_emissive_texture_path,
        TextureLoadFlags::color
    );
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
    if (!m_transforms_dirty)
        return;
    m_transform_mask = gltf_texture_transform_mask(m_texture_transforms);
    const bool volume = m_thickness_factor > 0.f;
    const bool coat = m_clearcoat_factor > 0.f;
    const bool sheen = m_sheen_color_factor.x > 0.f || m_sheen_color_factor.y > 0.f || m_sheen_color_factor.z > 0.f;
    const bool iridescence = m_iridescence_factor > 0.f;
    m_has_surface_extensions = m_anisotropy_strength > 0.f || m_diffuse_transmission_factor > 0.f;
    m_sheen_offset = m_iridescence_offset = 0xffffffffu;
    m_clearcoat_offset = 0;
    m_transform_offset = 0;
    if (m_transform_mask == 0 && !volume && !coat && !sheen && !iridescence && !m_has_surface_extensions) {
        ensure_data_size(0);
    } else {
        auto layout = m_scene->render_module()->layout();
        auto type = layout->find_type_by_name("StructuredBuffer<GLTFTextureUVTransform>");
        FALCOR_CHECK(type, "Failed to find GLTFTextureUVTransform type");
        auto element_layout = layout->get_type_layout(type)->element_type_layout();
        FALCOR_CHECK_LE(element_layout->alignment(), 16);
        ref<const sgl::TypeLayoutReflection> volume_layout;
        if (volume) {
            auto volume_type = layout->find_type_by_name("StructuredBuffer<GLTFVolumeData>");
            FALCOR_CHECK(volume_type, "Failed to find GLTFVolumeData type");
            volume_layout = layout->get_type_layout(volume_type)->element_type_layout();
            m_transform_offset = sgl::narrow_cast<uint32_t>(volume_layout->stride());
            FALCOR_CHECK(m_attenuation_distance > 0.f, "attenuation_distance must be positive");
        }
        ref<const sgl::TypeLayoutReflection> coat_layout;
        if (coat) {
            auto coat_type = layout->find_type_by_name("StructuredBuffer<GLTFClearcoatData>");
            FALCOR_CHECK(coat_type, "Failed to find GLTFClearcoatData type");
            coat_layout = layout->get_type_layout(coat_type)->element_type_layout();
            m_clearcoat_offset = m_transform_offset;
            m_transform_offset += sgl::narrow_cast<uint32_t>(coat_layout->stride());
        }
        ref<const sgl::TypeLayoutReflection> sheen_layout;
        if (sheen) {
            auto sheen_type = layout->find_type_by_name("StructuredBuffer<GLTFSheenData>");
            FALCOR_CHECK(sheen_type, "Failed to find GLTFSheenData type");
            sheen_layout = layout->get_type_layout(sheen_type)->element_type_layout();
            m_sheen_offset = m_transform_offset;
            m_transform_offset += sgl::narrow_cast<uint32_t>(sheen_layout->stride());
        }
        ref<const sgl::TypeLayoutReflection> iridescence_layout;
        if (iridescence) {
            auto iridescence_type = layout->find_type_by_name("StructuredBuffer<GLTFIridescenceData>");
            FALCOR_CHECK(iridescence_type, "Failed to find GLTFIridescenceData type");
            iridescence_layout = layout->get_type_layout(iridescence_type)->element_type_layout();
            m_iridescence_offset = m_transform_offset;
            m_transform_offset += sgl::narrow_cast<uint32_t>(iridescence_layout->stride());
        }
        ref<const sgl::TypeLayoutReflection> surface_layout;
        uint32_t surface_offset = m_transform_offset;
        if (m_has_surface_extensions) {
            auto surface_type = layout->find_type_by_name("StructuredBuffer<GLTFSurfaceExtensionsData>");
            FALCOR_CHECK(surface_type, "Failed to find GLTFSurfaceExtensionsData type");
            surface_layout = layout->get_type_layout(surface_type)->element_type_layout();
            FALCOR_CHECK_EQ(surface_layout->stride(), 40u);
            m_transform_offset += sgl::narrow_cast<uint32_t>(surface_layout->stride());
        }
        const size_t transform_bytes = std::popcount(m_transform_mask) * element_layout->stride();
        const size_t byte_size = m_transform_offset + transform_bytes;
        ensure_data_size(byte_size);
        auto bytes = mutable_data();
        if (volume) {
            float3 sigma_a{0.f};
            if (std::isfinite(m_attenuation_distance)) {
                for (int channel = 0; channel < 3; ++channel) {
                    const float color = std::clamp(m_attenuation_color[channel], 0.f, 1.f);
                    sigma_a[channel] = -std::log(color) / m_attenuation_distance;
                }
            }
            auto cursor = make_ref<sgl::BufferCursor>(
                ctx.device()->type(),
                volume_layout,
                bytes.data(),
                volume_layout->stride()
            );
            (*cursor)[0]["sigma_a"] = sigma_a;
        }
        if (coat) {
            auto cursor = make_ref<sgl::BufferCursor>(
                ctx.device()->type(),
                coat_layout,
                bytes.data() + m_clearcoat_offset,
                coat_layout->stride()
            );
            (*cursor)[0]["texture"] = m_clearcoat_texture_handle;
            (*cursor)[0]["roughness_texture"] = m_clearcoat_roughness_texture_handle;
            (*cursor)[0]["normal_texture"] = m_clearcoat_normal_texture_handle;
            (*cursor)[0]["normal_scale"] = float(m_clearcoat_normal_texture_scale);
            using Bits = shared::gltfmat::ClearcoatTCBits;
            uint32_t selectors = 0;
            Bits::texture::insert(selectors, m_clearcoat_texture_texcoord);
            Bits::roughness::insert(selectors, m_clearcoat_roughness_texture_texcoord);
            Bits::normal::insert(selectors, m_clearcoat_normal_texture_texcoord);
            Bits::normal_map_encoded::insert(
                selectors,
                uint32_t(is_normal_map_encoded(m_clearcoat_normal_texture_handle))
            );
            (*cursor)[0]["selectors"] = selectors;
        }
        if (sheen) {
            auto cursor = make_ref<sgl::BufferCursor>(
                ctx.device()->type(),
                sheen_layout,
                bytes.data() + m_sheen_offset,
                sheen_layout->stride()
            );
            (*cursor)[0]["color"] = m_sheen_color_factor;
            (*cursor)[0]["roughness"] = m_sheen_roughness_factor;
            (*cursor)[0]["color_texture"] = m_sheen_color_texture_handle;
            (*cursor)[0]["roughness_texture"] = m_sheen_roughness_texture_handle;
            using Bits = shared::gltfmat::SheenTCBits;
            uint32_t selectors = 0;
            Bits::color::insert(selectors, m_sheen_color_texture_texcoord);
            Bits::roughness::insert(selectors, m_sheen_roughness_texture_texcoord);
            (*cursor)[0]["selectors"] = selectors;
        }
        if (iridescence) {
            auto cursor = make_ref<sgl::BufferCursor>(
                ctx.device()->type(),
                iridescence_layout,
                bytes.data() + m_iridescence_offset,
                iridescence_layout->stride()
            );
            (*cursor)[0]["factor"] = m_iridescence_factor;
            (*cursor)[0]["ior"] = m_iridescence_ior;
            (*cursor)[0]["thickness_minimum"] = m_iridescence_thickness_minimum;
            (*cursor)[0]["thickness_maximum"] = m_iridescence_thickness_maximum;
            (*cursor)[0]["texture"] = m_iridescence_texture_handle;
            (*cursor)[0]["thickness_texture"] = m_iridescence_thickness_texture_handle;
            using Bits = shared::gltfmat::IridescenceTCBits;
            uint32_t selectors = 0;
            Bits::texture::insert(selectors, m_iridescence_texture_texcoord);
            Bits::thickness::insert(selectors, m_iridescence_thickness_texture_texcoord);
            (*cursor)[0]["selectors"] = selectors;
        }
        if (m_has_surface_extensions) {
            auto cursor = make_ref<sgl::BufferCursor>(
                ctx.device()->type(),
                surface_layout,
                bytes.data() + surface_offset,
                surface_layout->stride()
            );
            (*cursor)[0]["anisotropy_strength"] = m_anisotropy_strength;
            (*cursor)[0]["anisotropy_rotation"] = m_anisotropy_rotation;
            (*cursor)[0]["diffuse_transmission_factor"] = m_diffuse_transmission_factor;
            (*cursor)[0]["diffuse_transmission_color_factor"] = m_diffuse_transmission_color_factor;
            (*cursor)[0]["anisotropy_texture"] = m_anisotropy_texture_handle;
            (*cursor)[0]["diffuse_transmission_texture"] = m_diffuse_transmission_texture_handle;
            (*cursor)[0]["diffuse_transmission_color_texture"] = m_diffuse_transmission_color_texture_handle;
            using Bits = shared::gltfmat::SurfaceExtensionsTCBits;
            uint32_t selectors = 0;
            Bits::anisotropy::insert(selectors, m_anisotropy_texture_texcoord);
            Bits::diffuse_transmission::insert(selectors, m_diffuse_transmission_texture_texcoord);
            Bits::diffuse_transmission_color::insert(selectors, m_diffuse_transmission_color_texture_texcoord);
            (*cursor)[0]["selectors"] = selectors;
        }
        if (transform_bytes) {
            auto cursor = make_ref<sgl::BufferCursor>(
                ctx.device()->type(),
                element_layout,
                bytes.data() + m_transform_offset,
                transform_bytes
            );
            write_gltf_texture_transforms(*cursor, m_texture_transforms, m_scene->config().uv_origin);
        }
    }
    m_transforms_dirty = false;
}

shared::MaterialFlags GLTFMaterial::flags() const
{
    shared::MaterialFlags result = shared::MaterialFlags::none;
    if (m_double_sided || m_thickness_factor > 0.f)
        result |= shared::MaterialFlags::two_sided;
    if (m_thickness_factor <= 0.f)
        result |= shared::MaterialFlags::thin_walled;
    return result;
}

Material::OpacityDesc GLTFMaterial::opacity_desc() const
{
    shared::OpacityFlags opacity_flags = shared::OpacityFlags::none;
    if (m_alpha_mode == AlphaMode::mask)
        opacity_flags = shared::OpacityFlags::enabled | shared::OpacityFlags::use_threshold;
    else if (m_alpha_mode == AlphaMode::blend)
        opacity_flags = shared::OpacityFlags::enabled;
    if (opacity_flags != shared::OpacityFlags::none
        && (m_base_color_texture_texcoord != 0 || m_use_vertex_color || !m_texture_transforms[0].is_identity()))
        opacity_flags |= shared::OpacityFlags::evaluate_material;

    return {
        .flags = opacity_flags,
        .texture_handle = m_base_color_texture_handle,
        .texture_channel = 3,
        .factor = m_alpha_factor,
        .threshold = m_alpha_cutoff,
    };
}

SceneCapabilities GLTFMaterial::required_capabilities() const
{
    bool uses_geometry_properties = m_use_vertex_color || m_anisotropy_texture_texcoord != 0
        || m_diffuse_transmission_texture_texcoord != 0 || m_diffuse_transmission_color_texture_texcoord != 0
        || m_sheen_color_texture_texcoord != 0 || m_sheen_roughness_texture_texcoord != 0
        || m_iridescence_texture_texcoord != 0 || m_iridescence_thickness_texture_texcoord != 0
        || m_clearcoat_texture_texcoord != 0 || m_clearcoat_roughness_texture_texcoord != 0
        || m_clearcoat_normal_texture_texcoord != 0 || m_transmission_texture_texcoord != 0
        || m_specular_texture_texcoord != 0 || m_specular_color_texture_texcoord != 0
        || m_base_color_texture_texcoord != 0 || m_metallic_roughness_texture_texcoord != 0
        || m_normal_texture_texcoord != 0 || m_emissive_texture_texcoord != 0;
    return material_capabilities(uses_geometry_properties);
}

template<typename CursorT>
void GLTFMaterial::write_to_cursor_impl(CursorT cursor) const
{
    cursor["sheen_offset"] = m_sheen_offset;
    cursor["iridescence_offset"] = m_iridescence_offset;
    cursor["clearcoat_factor"] = m_clearcoat_factor;
    cursor["clearcoat_roughness_factor"] = m_clearcoat_roughness_factor;
    cursor["clearcoat_offset"] = m_clearcoat_offset;
    cursor["transmission_texture"] = m_transmission_texture_handle;
    cursor["transmission_factor"] = m_transmission_factor;
    cursor["specular_texture"] = m_specular_texture_handle;
    cursor["specular_color_texture"] = m_specular_color_texture_handle;
    cursor["specular_factor"] = m_specular_factor;
    cursor["specular_color_factor"] = float16_t4(m_specular_color_factor, 0.f);
    cursor["transform_mask"] = m_transform_mask;
    cursor["transform_offset"] = m_transform_offset;
    cursor["texture_transform_data"] = data_ref();
    cursor["base_color_texture"] = m_base_color_texture_handle;
    cursor["base_color_factor"] = float16_t4(m_base_color_factor, 0.f);
    cursor["metallic_roughness_texture"] = m_metallic_roughness_texture_handle;
    cursor["metallic_factor"] = m_metallic_factor;
    cursor["roughness_factor"] = m_roughness_factor;
    cursor["normal_texture"] = m_normal_texture_handle;
    cursor["normal_texture_scale"] = m_normal_texture_scale;
    cursor["emissive_factor"] = m_emissive_factor;
    cursor["emissive_texture"] = m_emissive_texture_handle;
    cursor["ior"] = m_ior;
    uint32_t selectors = pack_texture_coordinate_selectors(
        m_base_color_texture_texcoord,
        m_metallic_roughness_texture_texcoord,
        m_normal_texture_texcoord,
        m_emissive_texture_texcoord,
        m_specular_texture_texcoord,
        m_specular_color_texture_texcoord,
        m_transmission_texture_texcoord,
        is_normal_map_encoded(m_normal_texture_handle),
        m_use_vertex_color
    );
    shared::gltfmat::TCBits::has_surface_extensions::insert(selectors, uint32_t(m_has_surface_extensions));
    cursor["texture_coordinate_selectors"] = selectors;
}

template void GLTFMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void GLTFMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_MATERIAL(GLTFMaterial);

} // namespace falcor
