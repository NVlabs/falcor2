// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "gltf_specialization.h"
#include "gltf_texture_binding.h"
#include "shared_gltf_texture_selectors.h"
#include "falcor2/render/texture_manager.h"
#include "falcor2/render/scene_types.h"

#include <sgl/device/sampler.h>

#include "falcor2/core/types.h"

#include <array>
#include <string>

namespace falcor {

/// Reflective glTF specular-glossiness material with emission and alpha coverage.
/// Physical transmission and volume are not supported.
class FALCOR_API GLTFSpecGlossMaterial : public Material {
    FALCOR_SCENE_OBJECT(GLTFSpecGlossMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    using EffectiveMasks = GLTFMasks;

    /// Return the feature masks computed by the most recent material update.
    EffectiveMasks effective_masks() const
    {
        return {m_feature_mask, m_texture_mask, m_transform_mask, m_nonzero_texcoord_mask};
    }

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

    /// Reflect this class. The registration order is part of the existing property API.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        static constexpr std::array<TextureRegistration, 4> textures{{
            {TextureSlot::diffuse,
             "diffuse_texture",
             "Diffuse texture (takes precedence over path)",
             "Diffuse texture path",
             "Texture-coordinate set used by the diffuse texture.",
             false},
            {TextureSlot::specular_glossiness,
             "specular_glossiness_texture",
             "Specular-glossiness texture (takes precedence over path)",
             "Specular-glossiness texture path",
             "Texture-coordinate set used by the specular-glossiness texture.",
             false},
            {TextureSlot::normal,
             "normal_texture",
             "Normal texture (takes precedence over path)",
             "Normal texture path",
             "Texture-coordinate set used by the normal texture.",
             false},
            {TextureSlot::emissive,
             "emissive_texture",
             "Emissive texture (takes precedence over path)",
             "Emissive texture path",
             "Texture-coordinate set used by the emissive texture.",
             true},
        }};

        for (const auto& texture : textures)
            reflect_texture_transform(r, texture);

        r.def_prop_rw(
            "alpha_mode",
            [](const GLTFSpecGlossMaterial& material)
            {
                return material.m_core.alpha_mode;
            },
            [](GLTFSpecGlossMaterial& material, AlphaMode value)
            {
                material.m_core.alpha_mode = value;
            },
            "Alpha handling mode.",
            reflection::default_value(AlphaMode::opaque),
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
        );
        r.def_prop_rw(
            "alpha_factor",
            [](const GLTFSpecGlossMaterial& material)
            {
                return material.m_core.alpha_factor;
            },
            [](GLTFSpecGlossMaterial& material, float value)
            {
                material.m_core.alpha_factor = value;
            },
            "Constant alpha factor.",
            reflection::default_value(1.f),
            reflection::value_range_unit(),
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
        );
        r.def_prop_rw(
            "alpha_cutoff",
            [](const GLTFSpecGlossMaterial& material)
            {
                return material.m_core.alpha_cutoff;
            },
            [](GLTFSpecGlossMaterial& material, float value)
            {
                material.m_core.alpha_cutoff = value;
            },
            "Alpha cutoff used for mask mode.",
            reflection::default_value(0.5f),
            reflection::value_range_unit(),
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
        );
        r.def_prop_rw(
            "use_vertex_color",
            [](const GLTFSpecGlossMaterial& material)
            {
                return material.m_core.use_vertex_color;
            },
            [](GLTFSpecGlossMaterial& material, bool value)
            {
                material.m_core.use_vertex_color = value;
            },
            "Multiply diffuse and opacity by COLOR_0 when available.",
            reflection::default_value(false),
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
        );

        reflect_texture_binding(r, textures[0]);
        r.def_prop_rw(
            "diffuse_factor",
            [](const GLTFSpecGlossMaterial& material)
            {
                return material.m_core.diffuse_factor;
            },
            [](GLTFSpecGlossMaterial& material, float16_t3 value)
            {
                material.m_core.diffuse_factor = value;
            },
            "Diffuse factor",
            reflection::default_value(float16_t3{1.f, 1.f, 1.f}),
            reflection::value_range_unit(),
            reflection::UIFlags::display_as_color,
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
        );

        reflect_texture_binding(r, textures[1]);
        r.def_prop_rw(
            "specular_factor",
            [](const GLTFSpecGlossMaterial& material)
            {
                return material.m_core.specular_factor;
            },
            [](GLTFSpecGlossMaterial& material, float16_t3 value)
            {
                material.m_core.specular_factor = value;
            },
            "Linear RGB reflectance at normal incidence (F0).",
            reflection::default_value(float16_t3(1.f)),
            reflection::value_range_unit(),
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
        );
        r.def_prop_rw(
            "glossiness_factor",
            [](const GLTFSpecGlossMaterial& material)
            {
                return material.m_core.glossiness_factor;
            },
            [](GLTFSpecGlossMaterial& material, float16_t value)
            {
                material.m_core.glossiness_factor = value;
            },
            "Linear glossiness, multiplied by texture alpha; roughness is one minus glossiness.",
            reflection::default_value(float16_t(1.f)),
            reflection::value_range_unit(),
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
        );

        reflect_texture_binding(r, textures[2]);
        r.def_prop_rw(
            "normal_texture_scale",
            [](const GLTFSpecGlossMaterial& material)
            {
                return material.m_core.normal_texture_scale;
            },
            [](GLTFSpecGlossMaterial& material, float16_t value)
            {
                material.m_core.normal_texture_scale = value;
            },
            "Normal texture scale",
            reflection::default_value(float16_t(1.f)),
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
        );

        reflect_texture_binding(r, textures[3]);
        r.def_prop_rw(
            "emissive_factor",
            [](const GLTFSpecGlossMaterial& material)
            {
                return material.m_core.emissive_factor;
            },
            [](GLTFSpecGlossMaterial& material, float3 value)
            {
                material.m_core.emissive_factor = value;
            },
            "Emissive factor",
            reflection::default_value(float3(0.f)),
            reflection::value_range_positive(),
            reflection::UIFlags::display_as_color,
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_emission_properties)
        );
        r.def_prop_rw(
            "double_sided",
            [](const GLTFSpecGlossMaterial& material)
            {
                return material.m_core.double_sided;
            },
            [](GLTFSpecGlossMaterial& material, bool value)
            {
                material.m_core.double_sided = value;
            },
            "Double-sided material",
            reflection::default_value(false),
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_properties)
        );
        r.def_rw(
            "feature_mask_filter",
            &GLTFSpecGlossMaterial::m_feature_mask_filter,
            "Allowed glTF features. Clearing a bit disables its shader work without changing authored values.",
            reflection::default_value(shared::gltfmat::FeatureBits::ALL_SPEC_GLOSS),
            reflection::on_change(&GLTFSpecGlossMaterial::mark_dirty_emission_properties)
        );
    }

private:
    friend class GLTFContext;

    void set_compile_masks(EffectiveMasks masks);

    enum class TextureSlot : size_t {
        diffuse,
        specular_glossiness,
        normal,
        emissive,
        count,
    };

    struct TextureRegistration {
        TextureSlot slot;
        const char* property_prefix;
        const char* object_doc;
        const char* path_doc;
        const char* texcoord_doc;
        bool emission;
    };

    /// Authored scalar inputs for this workflow. Texture inputs live beside these
    /// in m_textures so each slot owns its sampler, UV selector, and transform.
    struct CoreInputs {
        AlphaMode alpha_mode{AlphaMode::opaque};
        float alpha_factor{1.f};
        float alpha_cutoff{0.5f};
        bool use_vertex_color{false};
        float16_t3 diffuse_factor{1.f, 1.f, 1.f};
        float16_t3 specular_factor{1.f};
        float16_t glossiness_factor{1.f};
        float16_t normal_texture_scale{1.f};
        float3 emissive_factor{0.f, 0.f, 0.f};
        bool double_sided{false};
    };

    GLTFTextureBinding& texture_binding(TextureSlot slot) { return m_textures[static_cast<size_t>(slot)]; }
    const GLTFTextureBinding& texture_binding(TextureSlot slot) const { return m_textures[static_cast<size_t>(slot)]; }

    void mark_texture_transform_dirty(bool emission)
    {
        if (emission)
            mark_dirty_emission_properties();
        else
            mark_dirty_properties();
    }

    void mark_texture_resource_dirty(bool emission)
    {
        if (emission)
            mark_dirty_emission_resources();
        else
            mark_dirty_resources();
    }

    void set_texture_texcoord(uint32_t& destination, uint32_t value, const char* property_name)
    {
        FALCOR_CHECK(value < 8, "{} must be in the range [0, 7], got {}.", property_name, value);
        if (destination != value) {
            destination = value;
            mark_dirty_properties();
        }
    }

    template<reflection::ClassReflector R>
    static void reflect_texture_transform(R& r, const TextureRegistration& texture)
    {
        const auto slot = texture.slot;
        const auto emission = texture.emission;
        const std::string offset_name = std::string(texture.property_prefix) + "_offset";
        const std::string rotation_name = std::string(texture.property_prefix) + "_rotation";
        const std::string scale_name = std::string(texture.property_prefix) + "_uv_scale";

        r.def_prop_rw(
            offset_name.c_str(),
            [slot](const GLTFSpecGlossMaterial& material)
            {
                return material.texture_binding(slot).transform.offset;
            },
            [slot, emission](GLTFSpecGlossMaterial& material, float2 value)
            {
                material.texture_binding(slot).transform.offset = value;
                material.mark_texture_transform_dirty(emission);
            },
            "UV offset in glTF upper-left coordinates.",
            reflection::default_value(float2(0.f)),
            reflection::UIFlags::advanced
        );
        r.def_prop_rw(
            rotation_name.c_str(),
            [slot](const GLTFSpecGlossMaterial& material)
            {
                return material.texture_binding(slot).transform.rotation;
            },
            [slot, emission](GLTFSpecGlossMaterial& material, float value)
            {
                material.texture_binding(slot).transform.rotation = value;
                material.mark_texture_transform_dirty(emission);
            },
            "UV rotation in radians in glTF upper-left coordinates.",
            reflection::default_value(0.f),
            reflection::UIFlags::advanced
        );
        r.def_prop_rw(
            scale_name.c_str(),
            [slot](const GLTFSpecGlossMaterial& material)
            {
                return material.texture_binding(slot).transform.scale;
            },
            [slot, emission](GLTFSpecGlossMaterial& material, float2 value)
            {
                material.texture_binding(slot).transform.scale = value;
                material.mark_texture_transform_dirty(emission);
            },
            "UV scale applied before rotation and offset.",
            reflection::default_value(float2(1.f)),
            reflection::UIFlags::advanced
        );
    }

    template<reflection::ClassReflector R>
    static void reflect_texture_binding(R& r, const TextureRegistration& texture)
    {
        const auto slot = texture.slot;
        const auto emission = texture.emission;
        const std::string object_name = texture.property_prefix;
        const std::string sampler_name = object_name + "_sampler";
        const std::string path_name = object_name + "_path";
        const std::string texcoord_name = object_name + "_texcoord";

        r.def_prop_rw(
            object_name.c_str(),
            [slot](const GLTFSpecGlossMaterial& material)
            {
                return material.texture_binding(slot).texture;
            },
            [slot, emission](GLTFSpecGlossMaterial& material, const ref<sgl::Texture>& value)
            {
                material.texture_binding(slot).texture = value;
                material.mark_texture_resource_dirty(emission);
            },
            texture.object_doc,
            reflection::UIFlags::advanced
        );
        r.def_prop_rw(
            sampler_name.c_str(),
            [slot](const GLTFSpecGlossMaterial& material)
            {
                return material.texture_binding(slot).sampler;
            },
            [slot, emission](GLTFSpecGlossMaterial& material, const ref<sgl::Sampler>& value)
            {
                material.texture_binding(slot).sampler = value;
                material.mark_texture_resource_dirty(emission);
            },
            "Texture sampler (null uses the scene default)",
            reflection::UIFlags::advanced
        );
        r.def_prop_rw(
            path_name.c_str(),
            [slot](const GLTFSpecGlossMaterial& material)
            {
                return material.texture_binding(slot).path;
            },
            [slot, emission](GLTFSpecGlossMaterial& material, const std::filesystem::path& value)
            {
                material.texture_binding(slot).path = value;
                material.mark_texture_resource_dirty(emission);
            },
            texture.path_doc,
            reflection::UIFlags::advanced
        );
        r.def_prop_rw(
            texcoord_name.c_str(),
            [slot](const GLTFSpecGlossMaterial& material)
            {
                return material.texture_binding(slot).effective_texcoord;
            },
            [slot, emission, texcoord_name](GLTFSpecGlossMaterial& material, uint32_t value)
            {
                auto& destination = material.texture_binding(slot).effective_texcoord;
                if (!emission || destination != value) {
                    material.set_texture_texcoord(destination, value, texcoord_name.c_str());
                    if (emission)
                        material.mark_dirty_emission_properties();
                }
            },
            texture.texcoord_doc,
            reflection::default_value(0u),
            reflection::value_range(0u, 7u),
            reflection::UIFlags::advanced
        );
    }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    CoreInputs m_core;
    std::array<GLTFTextureBinding, static_cast<size_t>(TextureSlot::count)> m_textures;
    uint32_t m_feature_mask{0};
    uint32_t m_feature_mask_filter{shared::gltfmat::FeatureBits::ALL_SPEC_GLOSS};
    uint32_t m_texture_mask{0};
    uint32_t m_transform_mask{0};
    uint32_t m_nonzero_texcoord_mask{0};
};

} // namespace falcor
