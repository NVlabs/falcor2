// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "falcor2/render/texture_manager.h"

#include "falcor2/core/types.h"

namespace falcor {

/// Standard specular-glossiness material.
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

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw(
                "diffuse_texture",
                &StandardSpecGlossMaterial::m_diffuse_texture,
                "Diffuse texture (takes precedence over path)",
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
                "specular_glossiness_texture_path",
                &StandardSpecGlossMaterial::m_specular_glossiness_texture_path,
                "Specular-glossiness texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
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
                "normal_texture_path",
                &StandardSpecGlossMaterial::m_normal_texture_path,
                "Normal texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
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
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "emissive_texture_path",
                &StandardSpecGlossMaterial::m_emissive_texture_path,
                "Emissive texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "emissive_factor",
                &StandardSpecGlossMaterial::m_emissive_factor,
                "Emissive factor",
                reflection::default_value(float3(0.f)),
                reflection::value_range_positive(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&StandardSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "transmission_texture",
                &StandardSpecGlossMaterial::m_transmission_texture,
                "Transmission texture (takes precedence over path)",
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
            );
    }

    // Accessors for Python Material.get_this() only.
    const TextureHandle& _diffuse_texture_handle() const { return m_diffuse_texture_handle; }
    float16_t4 _diffuse_factor() const { return float16_t4(m_diffuse_factor, 0.f); }
    const TextureHandle& _specular_glossiness_texture_handle() const { return m_specular_glossiness_texture_handle; }
    float16_t4 _specular_factor() const { return float16_t4(m_specular_factor, 0.f); }
    float16_t _glossiness_factor() const { return m_glossiness_factor; }
    const TextureHandle& _normal_texture_handle() const { return m_normal_texture_handle; }
    float16_t _normal_texture_scale() const { return m_normal_texture_scale; }
    float3 _emissive_factor() const { return m_emissive_factor; }
    const TextureHandle& _emissive_texture_handle() const { return m_emissive_texture_handle; }
    const TextureHandle& _transmission_texture_handle() const { return m_transmission_texture_handle; }
    float16_t _ior() const { return m_ior; }
    float16_t4 _transmission_factor() const { return float16_t4(m_transmission_factor, 0.f); }
    float16_t _diffuse_transmission_factor() const { return m_diffuse_transmission_factor; }
    float16_t _specular_transmission_factor() const { return m_specular_transmission_factor; }
    bool _double_sided() const { return m_double_sided; }
    bool _thin_walled() const { return m_thin_walled; }

private:
    void mark_dirty_resources() { mark_dirty(DirtyFlags::resources); }
    void mark_dirty_properties() { mark_dirty(DirtyFlags::properties); }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    // Static properties.
    ref<sgl::Texture> m_diffuse_texture;
    std::filesystem::path m_diffuse_texture_path;
    float16_t3 m_diffuse_factor{1.f, 1.f, 1.f};
    ref<sgl::Texture> m_specular_glossiness_texture;
    std::filesystem::path m_specular_glossiness_texture_path;
    float16_t3 m_specular_factor{1.f, 1.f, 1.f};
    float16_t m_glossiness_factor{1.f};
    ref<sgl::Texture> m_normal_texture;
    std::filesystem::path m_normal_texture_path;
    float16_t m_normal_texture_scale{1.f};
    ref<sgl::Texture> m_emissive_texture;
    std::filesystem::path m_emissive_texture_path;
    float3 m_emissive_factor{0.f, 0.f, 0.f};
    ref<sgl::Texture> m_transmission_texture;
    std::filesystem::path m_transmission_texture_path;
    float16_t m_ior{1.5f};
    float16_t3 m_transmission_factor{1.f, 1.f, 1.f};
    float16_t m_diffuse_transmission_factor{0.f};
    float16_t m_specular_transmission_factor{0.f};
    bool m_double_sided{false};
    bool m_thin_walled{false};

    TextureHandle m_diffuse_texture_handle;
    TextureHandle m_specular_glossiness_texture_handle;
    TextureHandle m_normal_texture_handle;
    TextureHandle m_emissive_texture_handle;
    TextureHandle m_transmission_texture_handle;
};

} // namespace falcor
