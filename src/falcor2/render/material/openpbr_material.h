// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "falcor2/render/texture_manager.h"

#include "falcor2/core/types.h"
#include "falcor2/core/reflection/dynamic_properties.h"

namespace falcor {

/// OpenPBR material.
class OpenPBRMaterial : public Material {
    FALCOR_SCENE_OBJECT(OpenPBRMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    /// Constructor.
    OpenPBRMaterial();

    /// Destructor.
    virtual ~OpenPBRMaterial() override;

    // ReflectedObject interface

    virtual void set_properties(const Properties& props) override;

    virtual reflection::DynamicPropertySet* dynamic_properties() override { return &m_material_properties; }
    virtual const reflection::DynamicPropertySet* dynamic_properties() const override { return &m_material_properties; }

    // SceneObject interface

    virtual void on_load_resources() override;

    // Material interface

    virtual void update(SceneUpdateContext& ctx) override;
    virtual shared::MaterialFlags flags() const override
    {
        return m_geometry_thin_walled ? shared::MaterialFlags::thin_walled : shared::MaterialFlags::none;
    }

public:
    enum class AttributeFlags {
        none = 0,
    };

private:
    void initialize_attributes();
    void initialize_properties();
    void sync_attributes_from_properties();

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    void require_update() { mark_dirty(DirtyFlags::properties); }
    void require_load_resources() { mark_dirty(DirtyFlags::resources); }

    enum class AttributeType {
        float_,
        float3,
    };

    enum class AttributeKind {
        /// Weight attribute in [0, 1] range.
        weight,
        /// Color attribute in [0, 1] range.
        color,
        /// Index of refraction in [1, +inf) range.
        ior,
        /// Generic attribute in [0, +inf) range.
        positive,
    };

    using AttributeValue = std::variant<float, float3>;

    struct AttributeInfo {
        std::string_view name;
        std::string_view slang_name;
        std::string_view label;
        std::string_view group;
        AttributeType type;
        AttributeKind kind;
        AttributeFlags flags;
        AttributeValue default_value;
    };

    struct Attribute {
        const AttributeInfo* info{nullptr};

        // Property values.
        AttributeValue value;
        ref<sgl::Texture> texture;
        std::filesystem::path texture_path;
        uint32_t texture_channel{0};

        // Runtime state.
        TextureHandle texture_handle;

        float get_float() const { return std::get<float>(value); }
        void set_float(float v) { value = v; }

        float3 get_float3() const { return std::get<float3>(value); }
        void set_float3(float3 v) { value = v; }
    };

    static constexpr auto GROUP_BASE = "Base";
    static constexpr auto GROUP_SUBSURFACE = "Subsurface";
    static constexpr auto GROUP_SPECULAR = "Specular";
    static constexpr auto GROUP_COAT = "Coat";
    static constexpr auto GROUP_FUZZ = "Fuzz";
    static constexpr auto GROUP_THIN_FILM = "Thin Film";
    static constexpr auto GROUP_EMISSION = "Emission";
    static constexpr auto GROUP_GEOMETRY = "Geometry";

    static constexpr std::array ATTRIBUTE_INFOS = {
        // clang-format off
        // Base
        AttributeInfo{"base_weight", "base_weight", "Weight", GROUP_BASE, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 1.f},
        AttributeInfo{"base_color", "base_color", "Color", GROUP_BASE, AttributeType::float3, AttributeKind::color, AttributeFlags::none, float3(0.8f)},
        AttributeInfo{"base_diffuse_roughness", "base_diffuse_roughness", "Diffuse Roughness", GROUP_BASE, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.f},
        AttributeInfo{"base_metalness", "base_metalness", "Metalness", GROUP_BASE, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.f},
        // Subsurface (TODO)
        // float subsurface_weight;
        // vec3 subsurface_color;
        // float subsurface_radius;
        // vec3 subsurface_radius_scale;
        // float subsurface_scatter_anisotropy;
        // Specular
        AttributeInfo{"specular_weight", "specular_weight", "Weight", GROUP_SPECULAR, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 1.f},
        AttributeInfo{"specular_color", "specular_color", "Color", GROUP_SPECULAR, AttributeType::float3, AttributeKind::color, AttributeFlags::none, float3(1.f)},
        AttributeInfo{"specular_roughness", "specular_roughness", "Roughness", GROUP_SPECULAR, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.3f},
        AttributeInfo{"specular_roughness_anisotropy", "specular_roughness_anisotropy", "Roughness Anisotropy", GROUP_SPECULAR, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.f},
        AttributeInfo{"specular_ior", "specular_ior", "Index of Refraction", GROUP_SPECULAR, AttributeType::float_, AttributeKind::ior, AttributeFlags::none, 1.5f},
        // specular_anisotropy_rotation_cos_sin
        // Coat
        AttributeInfo{"coat_weight", "coat_weight", "Weight", GROUP_COAT, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.f},
        AttributeInfo{"coat_color", "coat_color", "Color", GROUP_COAT, AttributeType::float3, AttributeKind::color, AttributeFlags::none, float3(1.f)},
        AttributeInfo{"coat_roughness", "coat_roughness", "Roughness", GROUP_COAT, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.f},
        AttributeInfo{"coat_roughness_anisotropy", "coat_roughness_anisotropy", "Roughness Anisotropy", GROUP_COAT, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.f},
        AttributeInfo{"coat_ior", "coat_ior", "Index of Refraction", GROUP_COAT, AttributeType::float_, AttributeKind::ior, AttributeFlags::none, 1.6f},
        AttributeInfo{"coat_darkening", "coat_darkening", "Darkening", GROUP_COAT, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.f},
        // coat_anisotropy_rotation_cos_sin
        // Fuzz
        AttributeInfo{"fuzz_weight", "fuzz_weight", "Weight", GROUP_FUZZ, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.f},
        AttributeInfo{"fuzz_color", "fuzz_color", "Color", GROUP_FUZZ, AttributeType::float3, AttributeKind::color, AttributeFlags::none, float3(1.f)},
        AttributeInfo{"fuzz_roughness", "fuzz_roughness", "Roughness", GROUP_FUZZ, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.5f},
        // Transmission / volume
        // float transmission_weight;
        // vec3 transmission_color;
        // float transmission_depth;
        // vec3 transmission_scatter;
        // float transmission_scatter_anisotropy;
        // float transmission_dispersion_scale;
        // float transmission_dispersion_abbe_number;
        // Thin-film
        AttributeInfo{"thin_film_weight", "thin_film_weight", "Weight", GROUP_THIN_FILM, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.f},
        AttributeInfo{"thin_film_thickness", "thin_film_thickness", "Thickness", GROUP_THIN_FILM, AttributeType::float_, AttributeKind::weight, AttributeFlags::none, 0.f},
        AttributeInfo{"thin_film_ior", "thin_film_ior", "Index of Refraction", GROUP_THIN_FILM, AttributeType::float_, AttributeKind::ior, AttributeFlags::none, 1.4f},
        // Emission
        AttributeInfo{"emission_luminance", "emission_luminance", "Luminance", GROUP_EMISSION, AttributeType::float_, AttributeKind::positive, AttributeFlags::none, 0.f},
        AttributeInfo{"emission_color", "emission_color", "Color", GROUP_EMISSION, AttributeType::float3, AttributeKind::color, AttributeFlags::none, float3(1.f)},
        // clang-format on
    };

    static constexpr size_t ATTRIBUTE_COUNT = ATTRIBUTE_INFOS.size();

    /// Dynamic properties (generated from attributes).
    reflection::DynamicPropertySet m_material_properties;

    ref<SceneGlobals> m_globals;

    std::array<Attribute, ATTRIBUTE_COUNT> m_attributes;
    ref<sgl::Buffer> m_attributes_buffer;

    ref<sgl::Texture> m_normal_texture;
    std::filesystem::path m_normal_texture_path;
    float m_normal_texture_scale{1.f};
    TextureHandle m_normal_texture_handle;
    bool m_geometry_thin_walled{false};

    const Attribute& attribute_by_name(std::string_view name) const;
};

FALCOR_ENUM_CLASS_OPERATORS(OpenPBRMaterial::AttributeFlags)

} // namespace falcor
