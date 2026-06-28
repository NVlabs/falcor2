// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "openpbr_material.h"

#include "falcor2/core/properties.h"

#include "falcor2/utils/buffer_handle.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/scene_globals.h"

// Include OpenPBR lookup tables.
#define OPENPBR_CONSTEXPR_GLOBAL static constexpr
#define OPENPBR_ENERGY_TABLES_USE_UINT16 1
#define OPENPBR_UINT16 uint16_t
namespace falcor::detail {
/// Small helper to convert the vec3 LUT data to float4, which is required by our texture format.
/// w component will be set to 0 and ignored in the shader.
struct vec3to4 {
    constexpr vec3to4(double x, double y, double z)
        : x(static_cast<float>(x))
        , y(static_cast<float>(y))
        , z(static_cast<float>(z))
        , w(0.f)
    {
    }
    float x, y, z, w;
};
} // namespace falcor::detail
#define vec3 falcor::detail::vec3to4
#include "../../external/openpbr-bsdf/impl/data/openpbr_energy_arrays.h"
#include "../../external/openpbr-bsdf/impl/data/openpbr_ltc_array.h"
#undef vec3

namespace falcor {

// ----------------------------------------------------------------------------
// OpenPBRSceneGlobals
// ----------------------------------------------------------------------------

/// Global shader data shared by OpenPBR materials.
/// This includes the precomputed lookup tables for the BRDF integration, which are stored as GPU textures.
class OpenPBRSceneGlobals : public SceneGlobals {
    FALCOR_OBJECT(OpenPBRSceneGlobals)
public:
    OpenPBRSceneGlobals(sgl::Device* device)
    {
        auto create_energy_lut = [device](std::string name, uint32_t dimensions, const void* data, size_t size) -> LUT
        {
            FALCOR_ASSERT(dimensions == 1 || dimensions == 2 || dimensions == 3);
            bool has_height = dimensions >= 2;
            bool has_depth = dimensions == 3;
            uint32_t dim = OpenPBR_EnergyTableSize;
            FALCOR_ASSERT(
                size == dim * (has_height ? dim : 1) * (has_depth ? dim : 1) * sizeof(OpenPBR_EnergyTableElement)
            );

            sgl::SubresourceData subresource_data = {
                .data = data,
                .size = size,
                .row_pitch = dim * sizeof(OpenPBR_EnergyTableElement),
                .slice_pitch = dim * dim * sizeof(OpenPBR_EnergyTableElement),
            };

            ref<sgl::Texture> texture = device->create_texture({
                .type = dimensions == 3 ? sgl::TextureType::texture_3d : sgl::TextureType::texture_2d,
                .format = sgl::Format::r16_unorm,
                .width = dim,
                .height = has_height ? dim : 1,
                .depth = has_depth ? dim : 1,
                .usage = sgl::TextureUsage::shader_resource,
                .label = name,
                .data = std::span{&subresource_data, 1},
            });

            sgl::DescriptorHandle descriptor_handle = texture->descriptor_handle_ro();

            return LUT{
                .name = std::move(name),
                .texture = std::move(texture),
                .descriptor_handle = descriptor_handle,
            };
        };

        auto create_ltc_lut = [device](std::string name, const void* data, size_t size) -> LUT
        {
            uint32_t dim = OpenPBR_LTCTableSize;
            FALCOR_ASSERT(size == dim * dim * sizeof(float4));

            sgl::SubresourceData subresource_data = {
                .data = data,
                .size = size,
                .row_pitch = dim * sizeof(float4),
                .slice_pitch = dim * dim * sizeof(float4),
            };

            ref<sgl::Texture> texture = device->create_texture({
                .type = sgl::TextureType::texture_2d,
                .format = sgl::Format::rgba32_float,
                .width = dim,
                .height = dim,
                .usage = sgl::TextureUsage::shader_resource,
                .label = name,
                .data = std::span{&subresource_data, 1},
            });

            sgl::DescriptorHandle descriptor_handle = texture->descriptor_handle_ro();

            return LUT{
                .name = std::move(name),
                .texture = std::move(texture),
                .descriptor_handle = descriptor_handle,
            };
        };

        // OpenPBR_LutId_IdealDielectricEnergyComplement (3D)
        m_luts[OpenPBR_LutId_IdealDielectricEnergyComplement] = create_energy_lut(
            "OpenPBR_IdealDielectricEnergyComplement",
            3,
            OpenPBR_IdealDielectricEnergyComplement_Array,
            sizeof(OpenPBR_IdealDielectricEnergyComplement_Array)
        );
        // OpenPBR_LutId_IdealDielectricAverageEnergyComplement (2D)
        m_luts[OpenPBR_LutId_IdealDielectricAverageEnergyComplement] = create_energy_lut(
            "OpenPBR_IdealDielectricAverageEnergyComplement",
            2,
            OpenPBR_IdealDielectricAverageEnergyComplement_Array,
            sizeof(OpenPBR_IdealDielectricAverageEnergyComplement_Array)
        );
        // OpenPBR_LutId_IdealDielectricReflectionRatio (2D)
        m_luts[OpenPBR_LutId_IdealDielectricReflectionRatio] = create_energy_lut(
            "OpenPBR_IdealDielectricReflectionRatio",
            2,
            OpenPBR_IdealDielectricReflectionRatio_Array,
            sizeof(OpenPBR_IdealDielectricReflectionRatio_Array)
        );
        // OpenPBR_LutId_OpaqueDielectricEnergyComplement (3D)
        m_luts[OpenPBR_LutId_OpaqueDielectricEnergyComplement] = create_energy_lut(
            "OpenPBR_OpaqueDielectricEnergyComplement",
            3,
            OpenPBR_OpaqueDielectricEnergyComplement_Array,
            sizeof(OpenPBR_OpaqueDielectricEnergyComplement_Array)
        );
        // OpenPBR_LutId_OpaqueDielectricAverageEnergyComplement (2D)
        m_luts[OpenPBR_LutId_OpaqueDielectricAverageEnergyComplement] = create_energy_lut(
            "OpenPBR_OpaqueDielectricAverageEnergyComplement",
            2,
            OpenPBR_OpaqueDielectricAverageEnergyComplement_Array,
            sizeof(OpenPBR_OpaqueDielectricAverageEnergyComplement_Array)
        );
        // OpenPBR_LutId_IdealMetalEnergyComplement (2D)
        m_luts[OpenPBR_LutId_IdealMetalEnergyComplement] = create_energy_lut(
            "OpenPBR_IdealMetalEnergyComplement",
            2,
            OpenPBR_IdealMetalEnergyComplement_Array,
            sizeof(OpenPBR_IdealMetalEnergyComplement_Array)
        );
        // OpenPBR_LutId_IdealMetalAverageEnergyComplement (2D)
        m_luts[OpenPBR_LutId_IdealMetalAverageEnergyComplement] = create_energy_lut(
            "OpenPBR_IdealMetalAverageEnergyComplement",
            1,
            OpenPBR_IdealMetalAverageEnergyComplement_Array,
            sizeof(OpenPBR_IdealMetalAverageEnergyComplement_Array)
        );
        // OpenPBR_LutId_LTC (2D)
        m_luts[OpenPBR_LutId_LTC] = create_ltc_lut("OpenPBR_LTC", OpenPBR_LTC_Array, sizeof(OpenPBR_LTC_Array));

        // Create sampler & descriptor handle for the sampler.
        m_sampler = device->create_sampler({
            .min_filter = sgl::TextureFilteringMode::linear,
            .mag_filter = sgl::TextureFilteringMode::linear,
            .mip_filter = sgl::TextureFilteringMode::linear,
            .address_u = sgl::TextureAddressingMode::clamp_to_edge,
            .address_v = sgl::TextureAddressingMode::clamp_to_edge,
            .address_w = sgl::TextureAddressingMode::clamp_to_edge,
            .label = "OpenPBR_LUT_Sampler",
        });
        m_sampler_handle = m_sampler->descriptor_handle();
    }

    virtual void bind(sgl::ShaderCursor globals) const override
    {
        auto openpbr_globals = globals.find_field("openpbr_globals");
        if (!openpbr_globals.is_valid())
            return;

        for (const auto& lut : m_luts)
            openpbr_globals[lut.name] = lut.descriptor_handle;
        openpbr_globals["sampler"] = m_sampler_handle;
    }

private:
    static constexpr size_t LUT_COUNT = 8;
    struct LUT {
        std::string name;
        ref<sgl::Texture> texture;
        sgl::DescriptorHandle descriptor_handle;
    };
    std::array<LUT, LUT_COUNT> m_luts;
    ref<sgl::Sampler> m_sampler;
    sgl::DescriptorHandle m_sampler_handle;
};

// ----------------------------------------------------------------------------
// OpenPBRMaterial
// ----------------------------------------------------------------------------

OpenPBRMaterial::OpenPBRMaterial()
{
    set_slang_type_name("OpenPBRMaterial");

    m_material_properties.set_instance(this);

    initialize_attributes();
    initialize_properties();
}

OpenPBRMaterial::~OpenPBRMaterial() { }

void OpenPBRMaterial::set_properties(const Properties& props)
{
    Material::set_properties(props);
}

void OpenPBRMaterial::on_load_resources()
{
    sync_attributes_from_properties();

    for (Attribute& attr : m_attributes) {
        if (attr.texture) {
            attr.texture_handle = m_scene->texture_manager()->register_texture({.texture = attr.texture});
        } else if (!attr.texture_path.empty()) {
            attr.texture_handle = m_scene->texture_manager()->load_texture({
                .path = attr.texture_path,
                .srgb = true,
                .load_deferred = true,
            });
        } else {
            attr.texture_handle = {};
        }
    }
    if (m_normal_texture) {
        m_normal_texture_handle = m_scene->texture_manager()->register_texture({.texture = m_normal_texture});
    } else if (!m_normal_texture_path.empty()) {
        m_normal_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_normal_texture_path,
            .srgb = false,
            .load_deferred = true,
        });
    }
}

void OpenPBRMaterial::update(SceneUpdateContext& ctx)
{
    if (!m_globals)
        m_globals = m_scene->get_or_create_scene_globals(
            "OpenPBRSceneGlobals",
            [device = ctx.device()]() -> ref<SceneGlobals>
            {
                return make_ref<OpenPBRSceneGlobals>(device);
            }
        );

    sync_attributes_from_properties();

    auto layout = m_scene->render_module()->layout();
    auto attributes_buffer_type = layout->find_type_by_name("StructuredBuffer<OpenPBRMaterial::Attributes>");
    FALCOR_CHECK(attributes_buffer_type, "Failed to find OpenPBRMaterial::Attributes type");

    auto attributes_layout = layout->get_type_layout(attributes_buffer_type)->element_type_layout();

    size_t attributes_size = attributes_layout->stride();
    auto attributes_data = std::make_unique<uint8_t[]>(attributes_size);

    auto buffer_cursor = make_ref<sgl::BufferCursor>(
        m_scene->device()->type(),
        attributes_layout,
        attributes_data.get(),
        attributes_size
    );

    auto attributes_cursor = (*buffer_cursor)[0];

    auto write_attribute = [&attributes_cursor](const Attribute& attr)
    {
        auto attribute_cursor = attributes_cursor[attr.info->slang_name];
        switch (attr.info->type) {
        case AttributeType::float_:
            attribute_cursor["value"] = static_cast<float16_t>(attr.get_float());
            attribute_cursor["texture_channel"] = static_cast<uint16_t>(attr.texture_channel);
            attribute_cursor["texture_handle"] = attr.texture_handle;
            break;
        case AttributeType::float3:
            attribute_cursor["value"] = static_cast<float16_t3>(attr.get_float3());
            attribute_cursor["texture_handle"] = attr.texture_handle;
            break;
        }
    };

    for (const Attribute& attr : m_attributes)
        write_attribute(attr);

    if (!m_attributes_buffer) {
        m_attributes_buffer = ctx.device()->create_buffer({
            .size = attributes_size,
            .usage = sgl::BufferUsage::shader_resource,
            .label = fmt::format("{}::attributes", m_name),
        });
    }

    ctx.command_encoder()->upload_buffer_data(m_attributes_buffer, 0, attributes_size, attributes_data.get());
}

template<typename CursorT>
void OpenPBRMaterial::write_to_cursor_impl(CursorT cursor) const
{
    // TODO(tdavidovic): This duplicates MaterialSystem's MaterialHeader.flags write. Reconcile the two paths once
    // direct material marshalling can get the same header data without this local write.
    cursor["header"]["flags"] = static_cast<uint32_t>(flags());
    cursor["normal_texture_handle"] = m_normal_texture_handle;
    cursor["normal_texture_scale"] = m_normal_texture_scale;
    cursor["attributes_buffer"] = to_handle(m_attributes_buffer);

    // TODO: We should probably safe this in the common material header flags instead.
    bool has_emission = attribute_by_name("emission_luminance").get_float() != 0.f
        && attribute_by_name("emission_color").get_float3() != float3(0.f);
    cursor["has_emission"] = has_emission;
}

template void OpenPBRMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void OpenPBRMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

void OpenPBRMaterial::initialize_attributes()
{
    for (size_t i = 0; i < ATTRIBUTE_COUNT; ++i) {
        auto& attr = m_attributes[i];
        attr.info = &ATTRIBUTE_INFOS[i];
        attr.value = attr.info->default_value;
    }
}

void OpenPBRMaterial::initialize_properties()
{
    auto value_range = [](const AttributeInfo& info) -> reflection::ValueRange
    {
        switch (info.kind) {
        case AttributeKind::weight:
            return reflection::value_range(0.0, 1.0);
        case AttributeKind::color:
            return reflection::value_range(0.0, 1.0);
        case AttributeKind::ior:
            return reflection::value_range(1.0, std::numeric_limits<float>::max());
        case AttributeKind::positive:
            return reflection::value_range(0.0, std::numeric_limits<float>::max());
        }
        return reflection::value_range(0.0, 1.0);
    };

    auto ui_flags = [](const AttributeInfo& info) -> reflection::UIFlags
    {
        switch (info.kind) {
        case AttributeKind::color:
            return reflection::UIFlags::display_as_color;
        default:
            return reflection::UIFlags::none;
        }
    };

    for (Attribute& attr : m_attributes) {
        const AttributeInfo& info = *attr.info;
        switch (info.type) {
        case AttributeType::float_:
            m_material_properties.add_property<float>(
                info.name,
                attr.get_float(),
                false,
                reflection::default_value(attr.get_float()),
                value_range(info),
                reflection::on_change(&OpenPBRMaterial::require_update),
                reflection::ui_label(info.label),
                reflection::ui_group(info.group),
                ui_flags(info)
            );
            break;
        case AttributeType::float3:
            m_material_properties.add_property<float3>(
                info.name,
                attr.get_float3(),
                false,
                reflection::default_value(attr.get_float3()),
                value_range(info),
                reflection::on_change(&OpenPBRMaterial::require_update),
                reflection::ui_label(info.label),
                reflection::ui_group(info.group),
                ui_flags(info)
            );
            break;
        }
        m_material_properties.add_property<ref<sgl::Texture>>(
            fmt::format("{}_texture", info.name),
            attr.texture,
            false,
            reflection::on_change(&OpenPBRMaterial::require_load_resources),
            reflection::ui_label(fmt::format("{} Texture", info.label)),
            reflection::ui_group(info.group),
            reflection::UIFlags::advanced

        );
        m_material_properties.add_property<std::filesystem::path>(
            fmt::format("{}_texture_path", info.name),
            attr.texture_path,
            false,
            reflection::on_change(&OpenPBRMaterial::require_load_resources),
            reflection::ui_label(fmt::format("{} Texture Path", info.label)),
            reflection::ui_group(info.group),
            reflection::UIFlags::advanced

        );
        if (info.type == AttributeType::float_) {
            m_material_properties.add_property<uint32_t>(
                fmt::format("{}_texture_channel", info.name),
                attr.texture_channel,
                false,
                reflection::on_change(&OpenPBRMaterial::require_update),
                reflection::ui_label(fmt::format("{} Texture Channel", info.label)),
                reflection::ui_group(info.group),
                reflection::UIFlags::advanced
            );
        }
    }

    m_material_properties.add_property<ref<sgl::Texture>>(
        "normal_texture",
        m_normal_texture,
        false,
        reflection::on_change(&OpenPBRMaterial::require_load_resources),
        reflection::ui_label("Normal Texture"),
        reflection::ui_group(GROUP_GEOMETRY),
        reflection::UIFlags::advanced
    );
    m_material_properties.add_property<std::filesystem::path>(
        "normal_texture_path",
        m_normal_texture_path,
        false,
        reflection::on_change(&OpenPBRMaterial::require_load_resources),
        reflection::ui_label("Normal Texture Path"),
        reflection::ui_group(GROUP_GEOMETRY),
        reflection::UIFlags::advanced
    );
    m_material_properties.add_property<float>(
        "normal_texture_scale",
        m_normal_texture_scale,
        false,
        reflection::default_value(m_normal_texture_scale),
        reflection::value_range(0.f, std::numeric_limits<float>::max()),
        reflection::on_change(&OpenPBRMaterial::require_update),
        reflection::ui_label("Normal Texture Scale"),
        reflection::ui_group(GROUP_GEOMETRY)
    );
    // https://academysoftwarefoundation.github.io/OpenPBR/#model/thin-walledmode
    m_material_properties.add_property<bool>(
        "geometry_thin_walled",
        m_geometry_thin_walled,
        false,
        reflection::default_value(m_geometry_thin_walled),
        reflection::on_change(&OpenPBRMaterial::require_update),
        reflection::ui_label("Thin Walled"),
        reflection::ui_group(GROUP_GEOMETRY),
        reflection::UIFlags::advanced
    );
}

void OpenPBRMaterial::sync_attributes_from_properties()
{
    for (Attribute& attr : m_attributes) {
        const AttributeInfo& info = *attr.info;
        switch (info.type) {
        case AttributeType::float_:
            attr.set_float(m_material_properties.get<float>(info.name));
            break;
        case AttributeType::float3:
            attr.set_float3(m_material_properties.get<float3>(info.name));
            break;
        }
        attr.texture = m_material_properties.get<ref<sgl::Texture>>(fmt::format("{}_texture", info.name));
        attr.texture_path = m_material_properties.get<std::filesystem::path>(fmt::format("{}_texture_path", info.name));
        if (info.type == AttributeType::float_) {
            attr.texture_channel = m_material_properties.get<uint32_t>(fmt::format("{}_texture_channel", info.name));
        }
    }

    m_normal_texture = m_material_properties.get<ref<sgl::Texture>>("normal_texture");
    m_normal_texture_path = m_material_properties.get<std::filesystem::path>("normal_texture_path");
    m_normal_texture_scale = m_material_properties.get<float>("normal_texture_scale");
    m_geometry_thin_walled = m_material_properties.get<bool>("geometry_thin_walled");
}

const OpenPBRMaterial::Attribute& OpenPBRMaterial::attribute_by_name(std::string_view name) const
{
    for (const Attribute& attr : m_attributes) {
        if (attr.info->name == name)
            return attr;
    }
    FALCOR_THROW("OpenPBR attribute with name \"{}\" not found", name);
}

FALCOR_SCENE_REGISTER_MATERIAL(OpenPBRMaterial);

} // namespace falcor
