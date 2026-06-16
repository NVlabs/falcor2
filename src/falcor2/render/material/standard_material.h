// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "falcor2/render/texture_manager.h"

#include "falcor2/core/types.h"

namespace falcor {

/// Standard material.
class FALCOR_API StandardMaterial : public Material {
    FALCOR_SCENE_OBJECT(StandardMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    /// Constructor.
    StandardMaterial();

    /// Destructor.
    virtual ~StandardMaterial() override;

    // SceneObject interface

    virtual void on_load_resources() override;

    // Material interface

    virtual void update(SceneUpdateContext& ctx) override;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw(
                "base_color_texture",
                &StandardMaterial::m_base_color_texture,
                "Base color texture (takes precedence over path)",
                reflection::on_change(&StandardMaterial::mark_dirty_resources)
            )
            .def_rw(
                "base_color_texture_path",
                &StandardMaterial::m_base_color_texture_path,
                "Base color texture path",
                reflection::on_change(&StandardMaterial::mark_dirty_resources)
            )
            .def_rw(
                "base_color_factor",
                &StandardMaterial::m_base_color_factor,
                "Base color factor",
                reflection::default_value(float3(1.f)),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "metallic_roughness_texture",
                &StandardMaterial::m_metallic_roughness_texture,
                "Metallic-roughness texture (takes precedence over path)",
                reflection::on_change(&StandardMaterial::mark_dirty_resources)
            )
            .def_rw(
                "metallic_roughness_texture_path",
                &StandardMaterial::m_metallic_roughness_texture_path,
                "Metallic-roughness texture path",
                reflection::on_change(&StandardMaterial::mark_dirty_resources)
            )
            .def_rw(
                "metallic_factor",
                &StandardMaterial::m_metallic_factor,
                "Metallic factor",
                reflection::default_value(0.f),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "roughness_factor",
                &StandardMaterial::m_roughness_factor,
                "Roughness factor",
                reflection::default_value(1.f),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "normal_texture",
                &StandardMaterial::m_normal_texture,
                "Normal texture (takes precedence over path)",
                reflection::on_change(&StandardMaterial::mark_dirty_resources)
            )
            .def_rw(
                "normal_texture_path",
                &StandardMaterial::m_normal_texture_path,
                "Normal texture path",
                reflection::on_change(&StandardMaterial::mark_dirty_resources)
            )
            .def_rw(
                "normal_texture_scale",
                &StandardMaterial::m_normal_texture_scale,
                "Normal texture scale",
                reflection::default_value(1.f),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "emissive_texture",
                &StandardMaterial::m_emissive_texture,
                "Emissive texture (takes precedence over path)",
                reflection::on_change(&StandardMaterial::mark_dirty_resources)
            )
            .def_rw(
                "emissive_texture_path",
                &StandardMaterial::m_emissive_texture_path,
                "Emissive texture path",
                reflection::on_change(&StandardMaterial::mark_dirty_resources)
            )
            .def_rw(
                "emissive_factor",
                &StandardMaterial::m_emissive_factor,
                "Emissive factor",
                reflection::default_value(float3(0.f)),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "transmission_texture",
                &StandardMaterial::m_transmission_texture,
                "Transmission texture (takes precedence over path)",
                reflection::on_change(&StandardMaterial::mark_dirty_resources)
            )
            .def_rw(
                "transmission_texture_path",
                &StandardMaterial::m_transmission_texture_path,
                "Transmission texture path",
                reflection::on_change(&StandardMaterial::mark_dirty_resources)
            )
            .def_rw(
                "ior",
                &StandardMaterial::m_ior,
                "Index of refraction",
                reflection::default_value(1.5f),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "transmission_factor",
                &StandardMaterial::m_transmission_factor,
                "Transmission factor",
                reflection::default_value(float3(1.f)),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "diffuse_transmission_factor",
                &StandardMaterial::m_diffuse_transmission_factor,
                "Diffuse transmission factor",
                reflection::default_value(0.f),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "specular_transmission_factor",
                &StandardMaterial::m_specular_transmission_factor,
                "Specular transmission factor",
                reflection::default_value(0.f),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "thin_surface",
                &StandardMaterial::m_thin_surface,
                "Thin surface",
                reflection::default_value(false),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "double_sided",
                &StandardMaterial::m_double_sided,
                "Double-sided material",
                reflection::default_value(false),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "metallic_texture_channel",
                &StandardMaterial::m_metallic_texture_channel,
                "Channel index for metallic value in the texture.",
                reflection::default_value(0u),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            )
            .def_rw(
                "roughness_texture_channel",
                &StandardMaterial::m_roughness_texture_channel,
                "Channel index for roughness value in the texture.",
                reflection::default_value(1u),
                reflection::on_change(&StandardMaterial::mark_dirty_properties)
            );
    }

    // Accessors for Python Material.get_this() only.
    const TextureHandle& _base_color_texture_handle() const { return m_base_color_texture_handle; }
    const float3& _base_color_factor() const { return m_base_color_factor; }
    const TextureHandle& _metallic_roughness_texture_handle() const { return m_metallic_roughness_texture_handle; }
    float _metallic_factor() const { return m_metallic_factor; }
    float _roughness_factor() const { return m_roughness_factor; }
    const TextureHandle& _normal_texture_handle() const { return m_normal_texture_handle; }
    float _normal_texture_scale() const { return m_normal_texture_scale; }
    const float3& _emissive_factor() const { return m_emissive_factor; }
    const TextureHandle& _emissive_texture_handle() const { return m_emissive_texture_handle; }
    const TextureHandle& _transmission_texture_handle() const { return m_transmission_texture_handle; }
    float _ior() const { return m_ior; }
    const float3& _transmission_factor() const { return m_transmission_factor; }
    float _diffuse_transmission_factor() const { return m_diffuse_transmission_factor; }
    float _specular_transmission_factor() const { return m_specular_transmission_factor; }
    bool _thin_surface() const { return m_thin_surface; }
    bool _double_sided() const { return m_double_sided; }
    uint _metallic_texture_channel() const { return m_metallic_texture_channel; }
    uint _roughness_texture_channel() const { return m_roughness_texture_channel; }

private:
    void mark_dirty_resources() { mark_dirty(DirtyFlags::resources); }
    void mark_dirty_properties() { mark_dirty(DirtyFlags::properties); }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    // Static properties.
    ref<sgl::Texture> m_base_color_texture;
    std::filesystem::path m_base_color_texture_path;
    float3 m_base_color_factor{1.f, 1.f, 1.f};
    ref<sgl::Texture> m_metallic_roughness_texture;
    std::filesystem::path m_metallic_roughness_texture_path;
    float m_metallic_factor{0.f};
    float m_roughness_factor{1.f};
    ref<sgl::Texture> m_normal_texture;
    std::filesystem::path m_normal_texture_path;
    float m_normal_texture_scale{1.f};
    ref<sgl::Texture> m_emissive_texture;
    std::filesystem::path m_emissive_texture_path;
    float3 m_emissive_factor{0.f, 0.f, 0.f};
    ref<sgl::Texture> m_transmission_texture;
    std::filesystem::path m_transmission_texture_path;
    float m_ior{1.5f};
    float3 m_transmission_factor{1.f, 1.f, 1.f};
    float m_diffuse_transmission_factor{0.f};
    float m_specular_transmission_factor{0.f};
    bool m_thin_surface{false};
    bool m_double_sided{false};
    uint m_metallic_texture_channel{0};
    uint m_roughness_texture_channel{1};

    TextureHandle m_base_color_texture_handle;
    TextureHandle m_normal_texture_handle;
    TextureHandle m_metallic_roughness_texture_handle;
    TextureHandle m_emissive_texture_handle;
    TextureHandle m_transmission_texture_handle;
};

} // namespace falcor
