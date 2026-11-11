// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "falcor2/render/texture_manager.h"
#include "falcor2/render/scene_types.h"

#include <sgl/device/sampler.h>

#include "falcor2/core/types.h"

namespace falcor {

/// Standard specular-glossiness material.
/// Diffuse and specular inputs follow glTF's linear diffuse-color and F0 convention.
/// Material setup derives the diffuse BSDF albedo by multiplying diffuse by (1 - max(specular)).
class FALCOR_API StandardSpecGlossMaterial : public Material {
    FALCOR_SCENE_OBJECT(StandardSpecGlossMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    /// Constructor.
    StandardSpecGlossMaterial();

    /// Destructor.
    virtual ~StandardSpecGlossMaterial() override;

    // SceneObject interface

    virtual void on_load_resources() override;

    // Material interface

    virtual void update(SceneUpdateContext& ctx) override;

    virtual shared::MaterialFlags flags() const override;

    virtual OpacityDesc opacity_desc() const override;

    virtual SceneCapabilities required_capabilities() const override;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw(
                "alpha_mode",
                &StandardSpecGlossMaterial::m_alpha_mode,
                "Alpha handling mode.",
                reflection::default_value(AlphaMode::opaque),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "alpha_factor",
                &StandardSpecGlossMaterial::m_alpha_factor,
                "Constant alpha factor.",
                reflection::default_value(1.f),
                reflection::value_range_unit(),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "alpha_cutoff",
                &StandardSpecGlossMaterial::m_alpha_cutoff,
                "Alpha cutoff used for mask mode.",
                reflection::default_value(0.5f),
                reflection::value_range_unit(),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "use_vertex_color",
                &StandardSpecGlossMaterial::m_use_vertex_color,
                "Multiply diffuse color and opacity by COLOR_0 when available.",
                reflection::default_value(false),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "diffuse_texture",
                &StandardSpecGlossMaterial::m_diffuse_texture,
                "Diffuse texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "diffuse_texture_sampler",
                &StandardSpecGlossMaterial::m_diffuse_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "diffuse_texture_path",
                &StandardSpecGlossMaterial::m_diffuse_texture_path,
                "Diffuse texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_texture_texcoord",
                &StandardSpecGlossMaterial::diffuse_texture_texcoord,
                &StandardSpecGlossMaterial::set_diffuse_texture_texcoord,
                "Texture-coordinate set used by the diffuse texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "diffuse_factor",
                &StandardSpecGlossMaterial::m_diffuse_factor,
                "Diffuse factor",
                reflection::default_value(float16_t3{1.f, 1.f, 1.f}),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "specular_glossiness_texture",
                &StandardSpecGlossMaterial::m_specular_glossiness_texture,
                "Specular-glossiness texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "specular_glossiness_texture_sampler",
                &StandardSpecGlossMaterial::m_specular_glossiness_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "specular_glossiness_texture_path",
                &StandardSpecGlossMaterial::m_specular_glossiness_texture_path,
                "Specular-glossiness texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_glossiness_texture_texcoord",
                &StandardSpecGlossMaterial::specular_glossiness_texture_texcoord,
                &StandardSpecGlossMaterial::set_specular_glossiness_texture_texcoord,
                "Texture-coordinate set used by the specular-glossiness texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "specular_factor",
                &StandardSpecGlossMaterial::m_specular_factor,
                "Specular factor",
                reflection::default_value(float16_t3{1.f, 1.f, 1.f}),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "glossiness_factor",
                &StandardSpecGlossMaterial::m_glossiness_factor,
                "Glossiness factor",
                reflection::default_value(float16_t(1.f)),
                reflection::value_range_unit(),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "normal_texture",
                &StandardSpecGlossMaterial::m_normal_texture,
                "Normal texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "normal_texture_sampler",
                &StandardSpecGlossMaterial::m_normal_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "normal_texture_path",
                &StandardSpecGlossMaterial::m_normal_texture_path,
                "Normal texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "normal_texture_texcoord",
                &StandardSpecGlossMaterial::normal_texture_texcoord,
                &StandardSpecGlossMaterial::set_normal_texture_texcoord,
                "Texture-coordinate set used by the normal texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "normal_texture_scale",
                &StandardSpecGlossMaterial::m_normal_texture_scale,
                "Normal texture scale",
                reflection::default_value(float16_t(1.f)),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "emissive_texture",
                &StandardSpecGlossMaterial::m_emissive_texture,
                "Emissive texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_emission_resources)
            )
            .def_rw(
                "emissive_texture_sampler",
                &StandardSpecGlossMaterial::m_emissive_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_emission_resources)
            )
            .def_rw(
                "emissive_texture_path",
                &StandardSpecGlossMaterial::m_emissive_texture_path,
                "Emissive texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_emission_resources)
            )
            .def_prop_rw(
                "emissive_texture_texcoord",
                &StandardSpecGlossMaterial::emissive_texture_texcoord,
                &StandardSpecGlossMaterial::set_emissive_texture_texcoord,
                "Texture-coordinate set used by the emissive texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "emissive_factor",
                &StandardSpecGlossMaterial::m_emissive_factor,
                "Emissive factor",
                reflection::default_value(float3(0.f)),
                reflection::value_range_positive(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_emission_properties)
            )
            .def_rw(
                "transmission_texture",
                &StandardSpecGlossMaterial::m_transmission_texture,
                "Transmission texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "transmission_texture_sampler",
                &StandardSpecGlossMaterial::m_transmission_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "transmission_texture_path",
                &StandardSpecGlossMaterial::m_transmission_texture_path,
                "Transmission texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "transmission_texture_texcoord",
                &StandardSpecGlossMaterial::transmission_texture_texcoord,
                &StandardSpecGlossMaterial::set_transmission_texture_texcoord,
                "Texture-coordinate set used by the transmission texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "ior",
                &StandardSpecGlossMaterial::m_ior,
                "Index of refraction",
                reflection::default_value(float16_t(1.5f)),
                reflection::value_range(1.0, 10.0),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "transmission_factor",
                &StandardSpecGlossMaterial::m_transmission_factor,
                "Transmission factor",
                reflection::default_value(float16_t3{1.f, 1.f, 1.f}),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "diffuse_transmission_factor",
                &StandardSpecGlossMaterial::m_diffuse_transmission_factor,
                "Diffuse transmission factor",
                reflection::default_value(float16_t(0.f)),
                reflection::value_range_unit(),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "specular_transmission_factor",
                &StandardSpecGlossMaterial::m_specular_transmission_factor,
                "Specular transmission factor",
                reflection::default_value(float16_t(0.f)),
                reflection::value_range_unit(),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "double_sided",
                &StandardSpecGlossMaterial::m_double_sided,
                "Double-sided material",
                reflection::default_value(false),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "thin_walled",
                &StandardSpecGlossMaterial::m_thin_walled,
                "Thin-walled material",
                reflection::default_value(false),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "volume_sigma_a",
                &StandardSpecGlossMaterial::m_volume_sigma_a,
                "Homogeneous volume absorption coefficient",
                reflection::default_value(float3(0.f)),
                reflection::value_range_positive(),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "volume_sigma_s",
                &StandardSpecGlossMaterial::m_volume_sigma_s,
                "Homogeneous volume scattering coefficient",
                reflection::default_value(float3(0.f)),
                reflection::value_range_positive(),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "volume_anisotropy",
                &StandardSpecGlossMaterial::m_volume_anisotropy,
                "Homogeneous volume Henyey-Greenstein anisotropy",
                reflection::default_value(0.f),
                reflection::value_range(-0.999, 0.999),
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            );
    }

private:
    void set_texture_texcoord(uint32_t& destination, uint32_t value, const char* property_name)
    {
        FALCOR_CHECK(value < 8, "{} must be in the range [0, 7], got {}.", property_name, value);
        if (destination != value) {
            destination = value;
            mark_dirty_properties();
        }
    }

    uint32_t diffuse_texture_texcoord() const { return m_diffuse_texture_texcoord; }
    void set_diffuse_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_diffuse_texture_texcoord, value, "diffuse_texture_texcoord");
    }
    uint32_t specular_glossiness_texture_texcoord() const { return m_specular_glossiness_texture_texcoord; }
    void set_specular_glossiness_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_specular_glossiness_texture_texcoord, value, "specular_glossiness_texture_texcoord");
    }
    uint32_t normal_texture_texcoord() const { return m_normal_texture_texcoord; }
    void set_normal_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_normal_texture_texcoord, value, "normal_texture_texcoord");
    }
    uint32_t emissive_texture_texcoord() const { return m_emissive_texture_texcoord; }
    void set_emissive_texture_texcoord(uint32_t value)
    {
        if (m_emissive_texture_texcoord != value) {
            set_texture_texcoord(m_emissive_texture_texcoord, value, "emissive_texture_texcoord");
            mark_dirty_emission_properties();
        }
    }
    uint32_t transmission_texture_texcoord() const { return m_transmission_texture_texcoord; }
    void set_transmission_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_transmission_texture_texcoord, value, "transmission_texture_texcoord");
    }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    // Static properties.
    AlphaMode m_alpha_mode{AlphaMode::opaque};
    float m_alpha_factor{1.f};
    float m_alpha_cutoff{0.5f};
    bool m_use_vertex_color{false};
    ref<sgl::Texture> m_diffuse_texture;
    ref<sgl::Sampler> m_diffuse_texture_sampler;
    std::filesystem::path m_diffuse_texture_path;
    float16_t3 m_diffuse_factor{1.f, 1.f, 1.f};
    ref<sgl::Texture> m_specular_glossiness_texture;
    ref<sgl::Sampler> m_specular_glossiness_texture_sampler;
    std::filesystem::path m_specular_glossiness_texture_path;
    float16_t3 m_specular_factor{1.f, 1.f, 1.f};
    float16_t m_glossiness_factor{1.f};
    ref<sgl::Texture> m_normal_texture;
    ref<sgl::Sampler> m_normal_texture_sampler;
    std::filesystem::path m_normal_texture_path;
    float16_t m_normal_texture_scale{1.f};
    ref<sgl::Texture> m_emissive_texture;
    ref<sgl::Sampler> m_emissive_texture_sampler;
    std::filesystem::path m_emissive_texture_path;
    float3 m_emissive_factor{0.f, 0.f, 0.f};
    ref<sgl::Texture> m_transmission_texture;
    ref<sgl::Sampler> m_transmission_texture_sampler;
    std::filesystem::path m_transmission_texture_path;
    float16_t m_ior{1.5f};
    float16_t3 m_transmission_factor{1.f, 1.f, 1.f};
    float16_t m_diffuse_transmission_factor{0.f};
    float16_t m_specular_transmission_factor{0.f};
    bool m_double_sided{false};
    bool m_thin_walled{false};
    float3 m_volume_sigma_a{0.f};
    float3 m_volume_sigma_s{0.f};
    float m_volume_anisotropy{0.f};
    uint32_t m_diffuse_texture_texcoord{0};
    uint32_t m_specular_glossiness_texture_texcoord{0};
    uint32_t m_normal_texture_texcoord{0};
    uint32_t m_emissive_texture_texcoord{0};
    uint32_t m_transmission_texture_texcoord{0};

    TextureHandle m_diffuse_texture_handle;
    TextureHandle m_specular_glossiness_texture_handle;
    TextureHandle m_normal_texture_handle;
    TextureHandle m_emissive_texture_handle;
    TextureHandle m_transmission_texture_handle;
};

} // namespace falcor
