// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "gltf_texture_transform.h"
#include "falcor2/render/texture_manager.h"
#include "falcor2/render/scene_types.h"

#include <sgl/device/sampler.h>

#include "falcor2/core/types.h"

namespace falcor {

/// Reflective glTF specular-glossiness material with emission and alpha coverage.
/// Physical transmission and volume are not supported.
class FALCOR_API GLTFSpecGlossMaterial : public Material {
    FALCOR_SCENE_OBJECT(GLTFSpecGlossMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    /// Constructor.
    GLTFSpecGlossMaterial();

    /// Destructor.
    virtual ~GLTFSpecGlossMaterial() override;

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
            .def_prop_rw(
                "diffuse_texture_offset",
                &GLTFSpecGlossMaterial::diffuse_texture_offset,
                &GLTFSpecGlossMaterial::set_diffuse_texture_offset,
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_texture_rotation",
                &GLTFSpecGlossMaterial::diffuse_texture_rotation,
                &GLTFSpecGlossMaterial::set_diffuse_texture_rotation,
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_texture_uv_scale",
                &GLTFSpecGlossMaterial::diffuse_texture_uv_scale,
                &GLTFSpecGlossMaterial::set_diffuse_texture_uv_scale,
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_glossiness_texture_offset",
                &GLTFSpecGlossMaterial::specular_glossiness_texture_offset,
                &GLTFSpecGlossMaterial::set_specular_glossiness_texture_offset,
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_glossiness_texture_rotation",
                &GLTFSpecGlossMaterial::specular_glossiness_texture_rotation,
                &GLTFSpecGlossMaterial::set_specular_glossiness_texture_rotation,
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_glossiness_texture_uv_scale",
                &GLTFSpecGlossMaterial::specular_glossiness_texture_uv_scale,
                &GLTFSpecGlossMaterial::set_specular_glossiness_texture_uv_scale,
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "normal_texture_offset",
                &GLTFSpecGlossMaterial::normal_texture_offset,
                &GLTFSpecGlossMaterial::set_normal_texture_offset,
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "normal_texture_rotation",
                &GLTFSpecGlossMaterial::normal_texture_rotation,
                &GLTFSpecGlossMaterial::set_normal_texture_rotation,
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "normal_texture_uv_scale",
                &GLTFSpecGlossMaterial::normal_texture_uv_scale,
                &GLTFSpecGlossMaterial::set_normal_texture_uv_scale,
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "emissive_texture_offset",
                &GLTFSpecGlossMaterial::emissive_texture_offset,
                &GLTFSpecGlossMaterial::set_emissive_texture_offset,
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "emissive_texture_rotation",
                &GLTFSpecGlossMaterial::emissive_texture_rotation,
                &GLTFSpecGlossMaterial::set_emissive_texture_rotation,
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "emissive_texture_uv_scale",
                &GLTFSpecGlossMaterial::emissive_texture_uv_scale,
                &GLTFSpecGlossMaterial::set_emissive_texture_uv_scale,
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "alpha_mode",
                &GLTFSpecGlossMaterial::m_alpha_mode,
                "Alpha handling mode.",
                reflection::default_value(AlphaMode::opaque),
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "alpha_factor",
                &GLTFSpecGlossMaterial::m_alpha_factor,
                "Constant alpha factor.",
                reflection::default_value(1.f),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "alpha_cutoff",
                &GLTFSpecGlossMaterial::m_alpha_cutoff,
                "Alpha cutoff used for mask mode.",
                reflection::default_value(0.5f),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "use_vertex_color",
                &GLTFSpecGlossMaterial::m_use_vertex_color,
                "Multiply diffuse and opacity by COLOR_0 when available.",
                reflection::default_value(false),
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "diffuse_texture",
                &GLTFSpecGlossMaterial::m_diffuse_texture,
                "Diffuse texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "diffuse_texture_sampler",
                &GLTFSpecGlossMaterial::m_diffuse_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "diffuse_texture_path",
                &GLTFSpecGlossMaterial::m_diffuse_texture_path,
                "Diffuse texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_texture_texcoord",
                &GLTFSpecGlossMaterial::diffuse_texture_texcoord,
                &GLTFSpecGlossMaterial::set_diffuse_texture_texcoord,
                "Texture-coordinate set used by the diffuse texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "diffuse_factor",
                &GLTFSpecGlossMaterial::m_diffuse_factor,
                "Diffuse factor",
                reflection::default_value(float16_t3{1.f, 1.f, 1.f}),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "specular_glossiness_texture",
                &GLTFSpecGlossMaterial::m_specular_glossiness_texture,
                "Specular-glossiness texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "specular_glossiness_texture_sampler",
                &GLTFSpecGlossMaterial::m_specular_glossiness_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "specular_glossiness_texture_path",
                &GLTFSpecGlossMaterial::m_specular_glossiness_texture_path,
                "Specular-glossiness texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_glossiness_texture_texcoord",
                &GLTFSpecGlossMaterial::specular_glossiness_texture_texcoord,
                &GLTFSpecGlossMaterial::set_specular_glossiness_texture_texcoord,
                "Texture-coordinate set used by the specular-glossiness texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "specular_factor",
                &GLTFSpecGlossMaterial::m_specular_factor,
                "Linear RGB reflectance at normal incidence (F0).",
                reflection::default_value(float16_t3(1.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "glossiness_factor",
                &GLTFSpecGlossMaterial::m_glossiness_factor,
                "Linear glossiness, multiplied by texture alpha; roughness is one minus glossiness.",
                reflection::default_value(float16_t(1.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "normal_texture",
                &GLTFSpecGlossMaterial::m_normal_texture,
                "Normal texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "normal_texture_sampler",
                &GLTFSpecGlossMaterial::m_normal_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_resources)
            )
            .def_rw(
                "normal_texture_path",
                &GLTFSpecGlossMaterial::m_normal_texture_path,
                "Normal texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "normal_texture_texcoord",
                &GLTFSpecGlossMaterial::normal_texture_texcoord,
                &GLTFSpecGlossMaterial::set_normal_texture_texcoord,
                "Texture-coordinate set used by the normal texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "normal_texture_scale",
                &GLTFSpecGlossMaterial::m_normal_texture_scale,
                "Normal texture scale",
                reflection::default_value(float16_t(1.f)),
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
            )
            .def_rw(
                "emissive_texture",
                &GLTFSpecGlossMaterial::m_emissive_texture,
                "Emissive texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_emission_resources)
            )
            .def_rw(
                "emissive_texture_sampler",
                &GLTFSpecGlossMaterial::m_emissive_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_emission_resources)
            )
            .def_rw(
                "emissive_texture_path",
                &GLTFSpecGlossMaterial::m_emissive_texture_path,
                "Emissive texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_emission_resources)
            )
            .def_prop_rw(
                "emissive_texture_texcoord",
                &GLTFSpecGlossMaterial::emissive_texture_texcoord,
                &GLTFSpecGlossMaterial::set_emissive_texture_texcoord,
                "Texture-coordinate set used by the emissive texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "emissive_factor",
                &GLTFSpecGlossMaterial::m_emissive_factor,
                "Emissive factor",
                reflection::default_value(float3(0.f)),
                reflection::value_range_positive(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_emission_properties)
            )

            .def_rw(
                "double_sided",
                &GLTFSpecGlossMaterial::m_double_sided,
                "Double-sided material",
                reflection::default_value(false),
                reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
            );
    }

private:
    float2 diffuse_texture_offset() const { return m_texture_transforms[0].offset; }
    void set_diffuse_texture_offset(float2 value)
    {
        m_texture_transforms[0].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float diffuse_texture_rotation() const { return m_texture_transforms[0].rotation; }
    void set_diffuse_texture_rotation(float value)
    {
        m_texture_transforms[0].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 diffuse_texture_uv_scale() const { return m_texture_transforms[0].scale; }
    void set_diffuse_texture_uv_scale(float2 value)
    {
        m_texture_transforms[0].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 specular_glossiness_texture_offset() const { return m_texture_transforms[1].offset; }
    void set_specular_glossiness_texture_offset(float2 value)
    {
        m_texture_transforms[1].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float specular_glossiness_texture_rotation() const { return m_texture_transforms[1].rotation; }
    void set_specular_glossiness_texture_rotation(float value)
    {
        m_texture_transforms[1].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 specular_glossiness_texture_uv_scale() const { return m_texture_transforms[1].scale; }
    void set_specular_glossiness_texture_uv_scale(float2 value)
    {
        m_texture_transforms[1].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 normal_texture_offset() const { return m_texture_transforms[2].offset; }
    void set_normal_texture_offset(float2 value)
    {
        m_texture_transforms[2].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float normal_texture_rotation() const { return m_texture_transforms[2].rotation; }
    void set_normal_texture_rotation(float value)
    {
        m_texture_transforms[2].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 normal_texture_uv_scale() const { return m_texture_transforms[2].scale; }
    void set_normal_texture_uv_scale(float2 value)
    {
        m_texture_transforms[2].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 emissive_texture_offset() const { return m_texture_transforms[3].offset; }
    void set_emissive_texture_offset(float2 value)
    {
        m_texture_transforms[3].offset = value;
        m_transforms_dirty = true;
        mark_dirty_emission_properties();
    }
    float emissive_texture_rotation() const { return m_texture_transforms[3].rotation; }
    void set_emissive_texture_rotation(float value)
    {
        m_texture_transforms[3].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_emission_properties();
    }
    float2 emissive_texture_uv_scale() const { return m_texture_transforms[3].scale; }
    void set_emissive_texture_uv_scale(float2 value)
    {
        m_texture_transforms[3].scale = value;
        m_transforms_dirty = true;
        mark_dirty_emission_properties();
    }

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

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    GLTFTextureTransform m_texture_transforms[4];
    uint32_t m_transform_mask{0};
    bool m_transforms_dirty{true};

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
    float16_t3 m_specular_factor{1.f};
    float16_t m_glossiness_factor{1.f};
    ref<sgl::Texture> m_normal_texture;
    ref<sgl::Sampler> m_normal_texture_sampler;
    std::filesystem::path m_normal_texture_path;
    float16_t m_normal_texture_scale{1.f};
    ref<sgl::Texture> m_emissive_texture;
    ref<sgl::Sampler> m_emissive_texture_sampler;
    std::filesystem::path m_emissive_texture_path;
    float3 m_emissive_factor{0.f, 0.f, 0.f};
    bool m_double_sided{false};
    uint32_t m_diffuse_texture_texcoord{0};
    uint32_t m_specular_glossiness_texture_texcoord{0};
    uint32_t m_normal_texture_texcoord{0};
    uint32_t m_emissive_texture_texcoord{0};

    TextureHandle m_diffuse_texture_handle;
    TextureHandle m_normal_texture_handle;
    TextureHandle m_specular_glossiness_texture_handle;
    TextureHandle m_emissive_texture_handle;
};

} // namespace falcor
