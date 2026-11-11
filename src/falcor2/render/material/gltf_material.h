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
#include <type_traits>
#include <utility>

namespace falcor {

// These wrappers keep the public property names and metadata visible while
// accessing fields grouped under feature structs. The actual accessors are
// checked by the C++ type system and shared by all reflection paths.
#define GLTF_FIELD(group, field)                                                                                       \
    &FieldProperty<&GLTFMaterial::group, &decltype(GLTFMaterial::group)::field>::get,                                  \
        &FieldProperty<&GLTFMaterial::group, &decltype(GLTFMaterial::group)::field>::set

#define GLTF_BINDING_FIELD(group, slot, field)                                                                         \
    &BindingFieldProperty<&GLTFMaterial::group, &decltype(GLTFMaterial::group)::slot, &GLTFTextureBinding::field>::    \
        get,                                                                                                           \
        &BindingFieldProperty<                                                                                         \
            &GLTFMaterial::group,                                                                                      \
            &decltype(GLTFMaterial::group)::slot,                                                                      \
            &GLTFTextureBinding::field>::set

#define GLTF_TEXCOORD(group, slot, name, extra, emission)                                                              \
    &TexcoordProperty<&GLTFMaterial::group, &decltype(GLTFMaterial::group)::slot>::get,                                \
        [](GLTFMaterial& self, uint32_t value)                                                                         \
    {                                                                                                                  \
        self.set_texture_texcoord((self.group).slot, value, #name, extra, emission);                                   \
    }

#define GLTF_TRANSFORM(group, slot, field, emission)                                                                   \
    &TransformProperty<                                                                                                \
        &GLTFMaterial::group,                                                                                          \
        &decltype(GLTFMaterial::group)::slot,                                                                          \
        &GLTFTextureTransform::field,                                                                                  \
        emission>::get,                                                                                                \
        &TransformProperty<                                                                                            \
            &GLTFMaterial::group,                                                                                      \
            &decltype(GLTFMaterial::group)::slot,                                                                      \
            &GLTFTextureTransform::field,                                                                              \
            emission>::set


/// glTF metallic-roughness material with thin or solid transmission and homogeneous absorption.
class FALCOR_API GLTFMaterial : public Material {
    FALCOR_SCENE_OBJECT(GLTFMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    using EffectiveMasks = GLTFMasks;

    /// Return the feature masks computed by the most recent material update.
    EffectiveMasks effective_masks() const
    {
        return {m_feature_mask, m_texture_mask, m_transform_mask, m_nonzero_texcoord_mask};
    }

    /// Constructor.
    GLTFMaterial();

    /// Destructor.
    virtual ~GLTFMaterial() override;

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
                "thickness_factor",
                GLTF_FIELD(m_volume, thickness_factor),
                "Zero selects a thin surface; positive values select a closed volume. Ray-traced distances determine "
                "absorption.",
                reflection::default_value(0.f),
                reflection::value_range_positive(),
                reflection::on_change(&GLTFMaterial::mark_dirty_volume)
            )
            .def_prop_rw(
                "attenuation_color",
                GLTF_FIELD(m_volume, attenuation_color),
                "Linear transmittance after travelling attenuation_distance inside the volume.",
                reflection::default_value(float3(1.f)),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&GLTFMaterial::mark_dirty_volume)
            )
            .def_prop_rw(
                "attenuation_distance",
                GLTF_FIELD(m_volume, attenuation_distance),
                "Positive attenuation distance in world units; infinity disables absorption.",
                reflection::default_value(std::numeric_limits<float>::infinity()),
                reflection::value_range_positive(),
                reflection::on_change(&GLTFMaterial::mark_dirty_volume)
            )
            .def_prop_rw(
                "transmission_factor",
                GLTF_FIELD(m_transmission, factor),
                "Fraction of the dielectric base transmitted through the surface; multiplied by texture red.",
                reflection::default_value(float16_t(0.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "anisotropy_texture",
                GLTF_BINDING_FIELD(m_anisotropy, texture, texture),
                "Linear data texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "anisotropy_texture_sampler",
                GLTF_BINDING_FIELD(m_anisotropy, texture, sampler),
                "Linear data texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "anisotropy_texture_path",
                GLTF_BINDING_FIELD(m_anisotropy, texture, path),
                "Linear data texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "anisotropy_texture_bytes",
                GLTF_BINDING_FIELD(m_anisotropy, texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "anisotropy_texture_texcoord",
                GLTF_TEXCOORD(m_anisotropy, texture, anisotropy_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "anisotropy_texture_offset",
                GLTF_TRANSFORM(m_anisotropy, texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "anisotropy_texture_rotation",
                GLTF_TRANSFORM(m_anisotropy, texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "anisotropy_texture_uv_scale",
                GLTF_TRANSFORM(m_anisotropy, texture, scale, false),
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_texture",
                GLTF_BINDING_FIELD(m_diffuse_transmission, texture, texture),
                "Linear data texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_transmission_texture_sampler",
                GLTF_BINDING_FIELD(m_diffuse_transmission, texture, sampler),
                "Linear data texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_transmission_texture_path",
                GLTF_BINDING_FIELD(m_diffuse_transmission, texture, path),
                "Linear data texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_transmission_texture_bytes",
                GLTF_BINDING_FIELD(m_diffuse_transmission, texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_transmission_texture_texcoord",
                GLTF_TEXCOORD(m_diffuse_transmission, texture, diffuse_transmission_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_texture_offset",
                GLTF_TRANSFORM(m_diffuse_transmission, texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_texture_rotation",
                GLTF_TRANSFORM(m_diffuse_transmission, texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_texture_uv_scale",
                GLTF_TRANSFORM(m_diffuse_transmission, texture, scale, false),
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture",
                GLTF_BINDING_FIELD(m_diffuse_transmission, color_texture, texture),
                "sRGB color texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_sampler",
                GLTF_BINDING_FIELD(m_diffuse_transmission, color_texture, sampler),
                "sRGB color texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_path",
                GLTF_BINDING_FIELD(m_diffuse_transmission, color_texture, path),
                "sRGB color texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_bytes",
                GLTF_BINDING_FIELD(m_diffuse_transmission, color_texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_texcoord",
                GLTF_TEXCOORD(
                    m_diffuse_transmission,
                    color_texture,
                    diffuse_transmission_color_texture_texcoord,
                    true,
                    false
                ),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_offset",
                GLTF_TRANSFORM(m_diffuse_transmission, color_texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_rotation",
                GLTF_TRANSFORM(m_diffuse_transmission, color_texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_uv_scale",
                GLTF_TRANSFORM(m_diffuse_transmission, color_texture, scale, false),
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "anisotropy_strength",
                GLTF_FIELD(m_anisotropy, strength),
                "Anisotropy strength; multiplied by texture blue.",
                reflection::default_value(0.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_prop_rw(
                "anisotropy_rotation",
                GLTF_FIELD(m_anisotropy, rotation),
                "Anisotropy direction rotation in radians.",
                reflection::default_value(0.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_prop_rw(
                "diffuse_transmission_factor",
                GLTF_FIELD(m_diffuse_transmission, factor),
                "Fraction of remaining dielectric diffuse energy transmitted; texture alpha.",
                reflection::default_value(0.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_prop_rw(
                "diffuse_transmission_color_factor",
                GLTF_FIELD(m_diffuse_transmission, color_factor),
                "Linear diffuse transmission color; multiplied by sRGB texture RGB.",
                reflection::default_value(float3(1.f)),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_prop_rw(
                "sheen_color_texture",
                GLTF_BINDING_FIELD(m_sheen, color_texture, texture),
                "sRGB sheen color texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "sheen_color_texture_sampler",
                GLTF_BINDING_FIELD(m_sheen, color_texture, sampler),
                "sRGB sheen color texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "sheen_color_texture_path",
                GLTF_BINDING_FIELD(m_sheen, color_texture, path),
                "sRGB sheen color texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "sheen_color_texture_bytes",
                GLTF_BINDING_FIELD(m_sheen, color_texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "sheen_color_texture_texcoord",
                GLTF_TEXCOORD(m_sheen, color_texture, sheen_color_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_color_texture_offset",
                GLTF_TRANSFORM(m_sheen, color_texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_color_texture_rotation",
                GLTF_TRANSFORM(m_sheen, color_texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_color_texture_uv_scale",
                GLTF_TRANSFORM(m_sheen, color_texture, scale, false),
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_roughness_texture",
                GLTF_BINDING_FIELD(m_sheen, roughness_texture, texture),
                "Linear alpha sheen roughness texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "sheen_roughness_texture_sampler",
                GLTF_BINDING_FIELD(m_sheen, roughness_texture, sampler),
                "Linear alpha sheen roughness texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "sheen_roughness_texture_path",
                GLTF_BINDING_FIELD(m_sheen, roughness_texture, path),
                "Linear alpha sheen roughness texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "sheen_roughness_texture_bytes",
                GLTF_BINDING_FIELD(m_sheen, roughness_texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "sheen_roughness_texture_texcoord",
                GLTF_TEXCOORD(m_sheen, roughness_texture, sheen_roughness_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_roughness_texture_offset",
                GLTF_TRANSFORM(m_sheen, roughness_texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_roughness_texture_rotation",
                GLTF_TRANSFORM(m_sheen, roughness_texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_roughness_texture_uv_scale",
                GLTF_TRANSFORM(m_sheen, roughness_texture, scale, false),
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_texture",
                GLTF_BINDING_FIELD(m_iridescence, texture, texture),
                "Linear red iridescence strength texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "iridescence_texture_sampler",
                GLTF_BINDING_FIELD(m_iridescence, texture, sampler),
                "Linear red iridescence strength texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "iridescence_texture_path",
                GLTF_BINDING_FIELD(m_iridescence, texture, path),
                "Linear red iridescence strength texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "iridescence_texture_bytes",
                GLTF_BINDING_FIELD(m_iridescence, texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "iridescence_texture_texcoord",
                GLTF_TEXCOORD(m_iridescence, texture, iridescence_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_texture_offset",
                GLTF_TRANSFORM(m_iridescence, texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_texture_rotation",
                GLTF_TRANSFORM(m_iridescence, texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_texture_uv_scale",
                GLTF_TRANSFORM(m_iridescence, texture, scale, false),
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_thickness_texture",
                GLTF_BINDING_FIELD(m_iridescence, thickness_texture, texture),
                "Linear green iridescence thickness texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "iridescence_thickness_texture_sampler",
                GLTF_BINDING_FIELD(m_iridescence, thickness_texture, sampler),
                "Linear green iridescence thickness texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "iridescence_thickness_texture_path",
                GLTF_BINDING_FIELD(m_iridescence, thickness_texture, path),
                "Linear green iridescence thickness texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "iridescence_thickness_texture_bytes",
                GLTF_BINDING_FIELD(m_iridescence, thickness_texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "iridescence_thickness_texture_texcoord",
                GLTF_TEXCOORD(m_iridescence, thickness_texture, iridescence_thickness_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_thickness_texture_offset",
                GLTF_TRANSFORM(m_iridescence, thickness_texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_thickness_texture_rotation",
                GLTF_TRANSFORM(m_iridescence, thickness_texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_thickness_texture_uv_scale",
                GLTF_TRANSFORM(m_iridescence, thickness_texture, scale, false),
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_color_factor",
                GLTF_FIELD(m_sheen, color_factor),
                "Linear sheen reflectance; zero disables the layer.",
                reflection::default_value(float3(0.f)),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_prop_rw(
                "sheen_roughness_factor",
                GLTF_FIELD(m_sheen, roughness_factor),
                "Perceptual sheen roughness, multiplied by texture alpha.",
                reflection::default_value(0.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_prop_rw(
                "iridescence_factor",
                GLTF_FIELD(m_iridescence, factor),
                "Thin-film Fresnel weight, multiplied by texture red.",
                reflection::default_value(0.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_prop_rw(
                "iridescence_ior",
                GLTF_FIELD(m_iridescence, ior),
                "Thin-film index of refraction (at least one).",
                reflection::default_value(1.3f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_prop_rw(
                "iridescence_thickness_minimum",
                GLTF_FIELD(m_iridescence, thickness_minimum),
                "Thickness in nanometers at texture green zero; may exceed maximum.",
                reflection::default_value(100.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_prop_rw(
                "iridescence_thickness_maximum",
                GLTF_FIELD(m_iridescence, thickness_maximum),
                "Thickness in nanometers at texture green one, or without a texture.",
                reflection::default_value(400.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_prop_rw(
                "clearcoat_texture",
                GLTF_BINDING_FIELD(m_clearcoat, texture, texture),
                "Linear red clearcoat texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_texture_sampler",
                GLTF_BINDING_FIELD(m_clearcoat, texture, sampler),
                "Linear red clearcoat texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_texture_path",
                GLTF_BINDING_FIELD(m_clearcoat, texture, path),
                "Linear red clearcoat texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_texture_bytes",
                GLTF_BINDING_FIELD(m_clearcoat, texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_texture_texcoord",
                GLTF_TEXCOORD(m_clearcoat, texture, clearcoat_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_texture_offset",
                GLTF_TRANSFORM(m_clearcoat, texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_texture_rotation",
                GLTF_TRANSFORM(m_clearcoat, texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_texture_uv_scale",
                GLTF_TRANSFORM(m_clearcoat, texture, scale, false),
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_roughness_texture",
                GLTF_BINDING_FIELD(m_clearcoat, roughness_texture, texture),
                "Linear green clearcoat roughness texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_sampler",
                GLTF_BINDING_FIELD(m_clearcoat, roughness_texture, sampler),
                "Linear green clearcoat roughness texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_path",
                GLTF_BINDING_FIELD(m_clearcoat, roughness_texture, path),
                "Linear green clearcoat roughness texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_bytes",
                GLTF_BINDING_FIELD(m_clearcoat, roughness_texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_texcoord",
                GLTF_TEXCOORD(m_clearcoat, roughness_texture, clearcoat_roughness_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_offset",
                GLTF_TRANSFORM(m_clearcoat, roughness_texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_rotation",
                GLTF_TRANSFORM(m_clearcoat, roughness_texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_uv_scale",
                GLTF_TRANSFORM(m_clearcoat, roughness_texture, scale, false),
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_normal_texture",
                GLTF_BINDING_FIELD(m_clearcoat, normal_texture, texture),
                "Tangent-space clearcoat normal texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_normal_texture_sampler",
                GLTF_BINDING_FIELD(m_clearcoat, normal_texture, sampler),
                "Tangent-space clearcoat normal texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_normal_texture_path",
                GLTF_BINDING_FIELD(m_clearcoat, normal_texture, path),
                "Tangent-space clearcoat normal texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_normal_texture_bytes",
                GLTF_BINDING_FIELD(m_clearcoat, normal_texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_normal_texture_texcoord",
                GLTF_TEXCOORD(m_clearcoat, normal_texture, clearcoat_normal_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_normal_texture_offset",
                GLTF_TRANSFORM(m_clearcoat, normal_texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_normal_texture_rotation",
                GLTF_TRANSFORM(m_clearcoat, normal_texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_normal_texture_uv_scale",
                GLTF_TRANSFORM(m_clearcoat, normal_texture, scale, false),
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_factor",
                GLTF_FIELD(m_clearcoat, factor),
                "Clearcoat intensity, multiplied by linear texture red.",
                reflection::default_value(float16_t(0.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_clearcoat)
            )
            .def_prop_rw(
                "clearcoat_roughness_factor",
                GLTF_FIELD(m_clearcoat, roughness_factor),
                "Clearcoat perceptual roughness, multiplied by linear texture green.",
                reflection::default_value(float16_t(0.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "clearcoat_normal_texture_scale",
                GLTF_FIELD(m_clearcoat, normal_texture_scale),
                "Scale of the clearcoat tangent-space normal X and Y components.",
                reflection::default_value(float16_t(1.f)),
                reflection::on_change(&GLTFMaterial::mark_dirty_clearcoat)
            )
            .def_prop_rw(
                "transmission_texture",
                GLTF_BINDING_FIELD(m_transmission, texture, texture),
                "Linear red transmission texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "transmission_texture_sampler",
                GLTF_BINDING_FIELD(m_transmission, texture, sampler),
                "Linear red transmission texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "transmission_texture_path",
                GLTF_BINDING_FIELD(m_transmission, texture, path),
                "Linear red transmission texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "transmission_texture_bytes",
                GLTF_BINDING_FIELD(m_transmission, texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "transmission_texture_texcoord",
                GLTF_TEXCOORD(m_transmission, texture, transmission_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "transmission_texture_offset",
                GLTF_TRANSFORM(m_transmission, texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "transmission_texture_rotation",
                GLTF_TRANSFORM(m_transmission, texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "transmission_texture_uv_scale",
                GLTF_TRANSFORM(m_transmission, texture, scale, false),
                "Authored glTF UV uv scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_texture",
                GLTF_BINDING_FIELD(m_specular, texture, texture),
                "Linear alpha specular strength texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_texture_sampler",
                GLTF_BINDING_FIELD(m_specular, texture, sampler),
                "Linear alpha specular strength texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_texture_path",
                GLTF_BINDING_FIELD(m_specular, texture, path),
                "Linear alpha specular strength texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_texture_bytes",
                GLTF_BINDING_FIELD(m_specular, texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_texture_texcoord",
                GLTF_TEXCOORD(m_specular, texture, specular_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_texture_offset",
                GLTF_TRANSFORM(m_specular, texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_texture_rotation",
                GLTF_TRANSFORM(m_specular, texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_texture_uv_scale",
                GLTF_TRANSFORM(m_specular, texture, scale, false),
                "Authored glTF UV uv scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_color_texture",
                GLTF_BINDING_FIELD(m_specular, color_texture, texture),
                "sRGB specular color texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_color_texture_sampler",
                GLTF_BINDING_FIELD(m_specular, color_texture, sampler),
                "sRGB specular color texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_color_texture_path",
                GLTF_BINDING_FIELD(m_specular, color_texture, path),
                "sRGB specular color texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_color_texture_bytes",
                GLTF_BINDING_FIELD(m_specular, color_texture, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_color_texture_texcoord",
                GLTF_TEXCOORD(m_specular, color_texture, specular_color_texture_texcoord, true, false),
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_color_texture_offset",
                GLTF_TRANSFORM(m_specular, color_texture, offset, false),
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_color_texture_rotation",
                GLTF_TRANSFORM(m_specular, color_texture, rotation, false),
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_color_texture_uv_scale",
                GLTF_TRANSFORM(m_specular, color_texture, scale, false),
                "Authored glTF UV uv scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_factor",
                GLTF_FIELD(m_specular, factor),
                "Dielectric specular strength, including grazing reflection; multiplied by texture alpha.",
                reflection::default_value(float16_t(1.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "specular_color_factor",
                GLTF_FIELD(m_specular, color_factor),
                "Linear dielectric F0 multiplier. Values above one are allowed; the resulting F0 is clamped.",
                reflection::default_value(float16_t3(1.f)),
                reflection::value_range_positive(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "base_color_texture_offset",
                GLTF_TRANSFORM(m_core, base_color, offset, false),
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "base_color_texture_rotation",
                GLTF_TRANSFORM(m_core, base_color, rotation, false),
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "base_color_texture_uv_scale",
                GLTF_TRANSFORM(m_core, base_color, scale, false),
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "metallic_roughness_texture_offset",
                GLTF_TRANSFORM(m_core, metallic_roughness, offset, false),
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "metallic_roughness_texture_rotation",
                GLTF_TRANSFORM(m_core, metallic_roughness, rotation, false),
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "metallic_roughness_texture_uv_scale",
                GLTF_TRANSFORM(m_core, metallic_roughness, scale, false),
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "normal_texture_offset",
                GLTF_TRANSFORM(m_core, normal, offset, false),
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "normal_texture_rotation",
                GLTF_TRANSFORM(m_core, normal, rotation, false),
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "normal_texture_uv_scale",
                GLTF_TRANSFORM(m_core, normal, scale, false),
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "emissive_texture_offset",
                GLTF_TRANSFORM(m_core, emissive, offset, true),
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "emissive_texture_rotation",
                GLTF_TRANSFORM(m_core, emissive, rotation, true),
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "emissive_texture_uv_scale",
                GLTF_TRANSFORM(m_core, emissive, scale, true),
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "alpha_mode",
                GLTF_FIELD(m_core, alpha_mode),
                "Alpha handling mode.",
                reflection::default_value(AlphaMode::opaque),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "alpha_factor",
                GLTF_FIELD(m_core, alpha_factor),
                "Constant alpha factor.",
                reflection::default_value(1.f),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "alpha_cutoff",
                GLTF_FIELD(m_core, alpha_cutoff),
                "Alpha cutoff used for mask mode.",
                reflection::default_value(0.5f),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "use_vertex_color",
                GLTF_FIELD(m_core, use_vertex_color),
                "Multiply base color and opacity by COLOR_0 when available.",
                reflection::default_value(false),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "base_color_texture",
                GLTF_BINDING_FIELD(m_core, base_color, texture),
                "Base color texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "base_color_texture_sampler",
                GLTF_BINDING_FIELD(m_core, base_color, sampler),
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "base_color_texture_path",
                GLTF_BINDING_FIELD(m_core, base_color, path),
                "Base color texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "base_color_texture_bytes",
                GLTF_BINDING_FIELD(m_core, base_color, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "base_color_texture_texcoord",
                GLTF_TEXCOORD(m_core, base_color, base_color_texture_texcoord, false, false),
                "Texture-coordinate set used by the base color texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "base_color_factor",
                GLTF_FIELD(m_core, base_color_factor),
                "Base color factor",
                reflection::default_value(float16_t3{1.f, 1.f, 1.f}),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "metallic_roughness_texture",
                GLTF_BINDING_FIELD(m_core, metallic_roughness, texture),
                "Metallic-roughness texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "metallic_roughness_texture_sampler",
                GLTF_BINDING_FIELD(m_core, metallic_roughness, sampler),
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "metallic_roughness_texture_path",
                GLTF_BINDING_FIELD(m_core, metallic_roughness, path),
                "Metallic-roughness texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "metallic_roughness_texture_bytes",
                GLTF_BINDING_FIELD(m_core, metallic_roughness, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "metallic_roughness_texture_texcoord",
                GLTF_TEXCOORD(m_core, metallic_roughness, metallic_roughness_texture_texcoord, false, false),
                "Texture-coordinate set used by the metallic-roughness texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "metallic_factor",
                GLTF_FIELD(m_core, metallic_factor),
                "Metallic factor (blue texture channel)",
                reflection::default_value(float16_t(1.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "roughness_factor",
                GLTF_FIELD(m_core, roughness_factor),
                "Perceptual roughness factor (green texture channel)",
                reflection::default_value(float16_t(1.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "normal_texture",
                GLTF_BINDING_FIELD(m_core, normal, texture),
                "Normal texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "normal_texture_sampler",
                GLTF_BINDING_FIELD(m_core, normal, sampler),
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "normal_texture_path",
                GLTF_BINDING_FIELD(m_core, normal, path),
                "Normal texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "normal_texture_bytes",
                GLTF_BINDING_FIELD(m_core, normal, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "normal_texture_texcoord",
                GLTF_TEXCOORD(m_core, normal, normal_texture_texcoord, false, false),
                "Texture-coordinate set used by the normal texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "normal_texture_scale",
                GLTF_FIELD(m_core, normal_texture_scale),
                "Normal texture scale",
                reflection::default_value(float16_t(1.f)),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "emissive_texture",
                GLTF_BINDING_FIELD(m_core, emissive, texture),
                "Emissive texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_emission_resources)
            )
            .def_prop_rw(
                "emissive_texture_sampler",
                GLTF_BINDING_FIELD(m_core, emissive, sampler),
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_emission_resources)
            )
            .def_prop_rw(
                "emissive_texture_path",
                GLTF_BINDING_FIELD(m_core, emissive, path),
                "Emissive texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_emission_resources)
            )
            .def_prop_rw(
                "emissive_texture_bytes",
                GLTF_BINDING_FIELD(m_core, emissive, bytes),
                "Encoded texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_emission_resources)
            )
            .def_prop_rw(
                "emissive_texture_texcoord",
                GLTF_TEXCOORD(m_core, emissive, emissive_texture_texcoord, false, true),
                "Texture-coordinate set used by the emissive texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "emissive_factor",
                GLTF_FIELD(m_core, emissive_factor),
                "Emissive factor",
                reflection::default_value(float3(0.f)),
                reflection::value_range_positive(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&GLTFMaterial::mark_dirty_emission_properties)
            )

            .def_prop_rw(
                "ior",
                GLTF_FIELD(m_core, ior),
                "Dielectric index of refraction for Fresnel and thin-transmission roughness",
                reflection::default_value(float16_t(1.5f)),
                reflection::value_range(0.0, 10.0),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )

            .def_prop_rw(
                "double_sided",
                GLTF_FIELD(m_core, double_sided),
                "Double-sided material",
                reflection::default_value(false),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "feature_mask_filter",
                &GLTFMaterial::m_feature_mask_filter,
                "Allowed glTF features. Clearing a bit disables its shader work without changing authored values.",
                reflection::default_value(shared::gltfmat::FeatureBits::ALL_MR),
                reflection::on_change(&GLTFMaterial::mark_dirty_emission_properties)
            );
    }


private:
    friend class GLTFContext;

    void set_compile_masks(EffectiveMasks masks);

    template<auto Group, auto Field>
    struct FieldProperty {
        using Value = std::remove_cvref_t<decltype((std::declval<GLTFMaterial&>().*Group).*Field)>;
        static Value get(const GLTFMaterial& self) { return (self.*Group).*Field; }
        static void set(GLTFMaterial& self, const Value& value) { (self.*Group).*Field = value; }
    };

    template<auto Group, auto Binding, auto Field>
    struct BindingFieldProperty {
        using Value = std::remove_cvref_t<decltype(((std::declval<GLTFMaterial&>().*Group).*Binding).*Field)>;
        static Value get(const GLTFMaterial& self) { return ((self.*Group).*Binding).*Field; }
        static void set(GLTFMaterial& self, const Value& value) { ((self.*Group).*Binding).*Field = value; }
    };

    template<auto Group, auto Binding>
    struct TexcoordProperty {
        static uint32_t get(const GLTFMaterial& self) { return ((self.*Group).*Binding).effective_texcoord; }
    };

    template<auto Group, auto Binding, auto Field, bool Emission>
    struct TransformProperty {
        using Value = std::remove_cvref_t<decltype(std::declval<GLTFTextureTransform>().*Field)>;
        static Value get(const GLTFMaterial& self) { return (((self.*Group).*Binding).transform).*Field; }
        static void set(GLTFMaterial& self, Value value)
        {
            self.set_texture_transform((self.*Group).*Binding, Field, value, Emission);
        }
    };

    // Authored factors and texture bindings are grouped by glTF feature. The
    // bindings are CPU state; the GPU stores only core values and effective masks.
    struct CoreInputs {
        GLTFTextureBinding base_color;
        GLTFTextureBinding metallic_roughness;
        GLTFTextureBinding normal;
        GLTFTextureBinding emissive;
        AlphaMode alpha_mode{AlphaMode::opaque};
        float alpha_factor{1.f};
        float alpha_cutoff{0.5f};
        bool use_vertex_color{false};
        float16_t3 base_color_factor{1.f, 1.f, 1.f};
        float16_t metallic_factor{1.f};
        float16_t roughness_factor{1.f};
        float16_t normal_texture_scale{1.f};
        float3 emissive_factor{0.f, 0.f, 0.f};
        float16_t ior{1.5f};
        bool double_sided{false};
    };

    struct VolumeInputs {
        float thickness_factor{0.f};
        float3 attenuation_color{1.f};
        float attenuation_distance{std::numeric_limits<float>::infinity()};
    };

    struct TransmissionInputs {
        GLTFTextureBinding texture;
        float16_t factor{0.f};
    };

    struct SpecularInputs {
        GLTFTextureBinding texture;
        GLTFTextureBinding color_texture;
        float16_t factor{1.f};
        float16_t3 color_factor{1.f};
    };

    struct ClearcoatInputs {
        GLTFTextureBinding texture;
        GLTFTextureBinding roughness_texture;
        GLTFTextureBinding normal_texture;
        float16_t factor{0.f};
        float16_t roughness_factor{0.f};
        float16_t normal_texture_scale{1.f};
    };

    struct SheenInputs {
        GLTFTextureBinding color_texture;
        GLTFTextureBinding roughness_texture;
        float3 color_factor{0.f};
        float roughness_factor{0.f};
    };

    struct IridescenceInputs {
        GLTFTextureBinding texture;
        GLTFTextureBinding thickness_texture;
        float factor{0.f};
        float ior{1.3f};
        float thickness_minimum{100.f};
        float thickness_maximum{400.f};
    };

    struct AnisotropyInputs {
        GLTFTextureBinding texture;
        float strength{0.f};
        float rotation{0.f};
    };

    struct DiffuseTransmissionInputs {
        GLTFTextureBinding texture;
        GLTFTextureBinding color_texture;
        float factor{0.f};
        float3 color_factor{1.f};
    };

    // This order is part of the existing GPU transform-mask contract.
    template<typename Self>
    static auto texture_bindings(Self& self)
    {
        return std::array{
            &self.m_core.base_color,
            &self.m_core.metallic_roughness,
            &self.m_core.normal,
            &self.m_core.emissive,
            &self.m_specular.texture,
            &self.m_specular.color_texture,
            &self.m_transmission.texture,
            &self.m_clearcoat.texture,
            &self.m_clearcoat.roughness_texture,
            &self.m_clearcoat.normal_texture,
            &self.m_sheen.color_texture,
            &self.m_sheen.roughness_texture,
            &self.m_iridescence.texture,
            &self.m_iridescence.thickness_texture,
            &self.m_anisotropy.texture,
            &self.m_diffuse_transmission.texture,
            &self.m_diffuse_transmission.color_texture,
        };
    }

    void set_texture_texcoord(
        GLTFTextureBinding& binding,
        uint32_t value,
        const char* property_name,
        [[maybe_unused]] bool affects_extra_data,
        bool affects_emission
    )
    {
        FALCOR_CHECK(value < 8, "{} must be in the range [0, 7], got {}.", property_name, value);
        if (binding.effective_texcoord != value) {
            binding.effective_texcoord = value;
            mark_dirty_properties();
            if (affects_emission)
                mark_dirty_emission_properties();
        }
    }

    template<typename V>
    void
    set_texture_transform(GLTFTextureBinding& binding, V GLTFTextureTransform::* field, V value, bool affects_emission)
    {
        binding.transform.*field = value;
        if (affects_emission)
            mark_dirty_emission_properties();
        else
            mark_dirty_properties();
    }

    void mark_dirty_clearcoat() { mark_dirty_properties(); }
    void mark_dirty_extensions() { mark_dirty_properties(); }
    void mark_dirty_volume()
    {
        // A live thin-to-solid edit must retain the exit boundary in traversal.
        if (m_volume.thickness_factor > 0.f)
            set_enable_backface_culling(false);
        mark_dirty_properties();
    }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    CoreInputs m_core;
    VolumeInputs m_volume;
    TransmissionInputs m_transmission;
    SpecularInputs m_specular;
    ClearcoatInputs m_clearcoat;
    SheenInputs m_sheen;
    IridescenceInputs m_iridescence;
    AnisotropyInputs m_anisotropy;
    DiffuseTransmissionInputs m_diffuse_transmission;

    // Effective masks are refreshed together on each dirty material update.
    uint32_t m_feature_mask{0};
    uint32_t m_feature_mask_filter{shared::gltfmat::FeatureBits::ALL_MR};
    uint32_t m_texture_mask{0};
    uint32_t m_transform_mask{0};
    uint32_t m_nonzero_texcoord_mask{0};
    ref<SceneGlobals> m_scattering_globals;
};

#undef GLTF_TRANSFORM
#undef GLTF_TEXCOORD
#undef GLTF_BINDING_FIELD
#undef GLTF_FIELD

} // namespace falcor
