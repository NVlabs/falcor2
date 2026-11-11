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

/// glTF metallic-roughness material with thin or solid transmission and homogeneous absorption.
class FALCOR_API GLTFMaterial : public Material {
    FALCOR_SCENE_OBJECT(GLTFMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
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
            .def_rw(
                "thickness_factor",
                &GLTFMaterial::m_thickness_factor,
                "Zero selects a thin surface; positive values select a closed volume. Ray-traced distances determine "
                "absorption.",
                reflection::default_value(0.f),
                reflection::value_range_positive(),
                reflection::on_change(&GLTFMaterial::mark_dirty_volume)
            )
            .def_rw(
                "attenuation_color",
                &GLTFMaterial::m_attenuation_color,
                "Linear transmittance after travelling attenuation_distance inside the volume.",
                reflection::default_value(float3(1.f)),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&GLTFMaterial::mark_dirty_volume)
            )
            .def_rw(
                "attenuation_distance",
                &GLTFMaterial::m_attenuation_distance,
                "Positive attenuation distance in world units; infinity disables absorption.",
                reflection::default_value(std::numeric_limits<float>::infinity()),
                reflection::value_range_positive(),
                reflection::on_change(&GLTFMaterial::mark_dirty_volume)
            )
            .def_rw(
                "transmission_factor",
                &GLTFMaterial::m_transmission_factor,
                "Fraction of the dielectric base transmitted through the surface; multiplied by texture red.",
                reflection::default_value(float16_t(0.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "anisotropy_texture",
                &GLTFMaterial::m_anisotropy_texture,
                "Linear data texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "anisotropy_texture_sampler",
                &GLTFMaterial::m_anisotropy_texture_sampler,
                "Linear data texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "anisotropy_texture_path",
                &GLTFMaterial::m_anisotropy_texture_path,
                "Linear data texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "anisotropy_texture_texcoord",
                &GLTFMaterial::anisotropy_texture_texcoord,
                &GLTFMaterial::set_anisotropy_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "anisotropy_texture_offset",
                &GLTFMaterial::anisotropy_texture_offset,
                &GLTFMaterial::set_anisotropy_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "anisotropy_texture_rotation",
                &GLTFMaterial::anisotropy_texture_rotation,
                &GLTFMaterial::set_anisotropy_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "anisotropy_texture_uv_scale",
                &GLTFMaterial::anisotropy_texture_uv_scale,
                &GLTFMaterial::set_anisotropy_texture_uv_scale,
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "diffuse_transmission_texture",
                &GLTFMaterial::m_diffuse_transmission_texture,
                "Linear data texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "diffuse_transmission_texture_sampler",
                &GLTFMaterial::m_diffuse_transmission_texture_sampler,
                "Linear data texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "diffuse_transmission_texture_path",
                &GLTFMaterial::m_diffuse_transmission_texture_path,
                "Linear data texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_transmission_texture_texcoord",
                &GLTFMaterial::diffuse_transmission_texture_texcoord,
                &GLTFMaterial::set_diffuse_transmission_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_texture_offset",
                &GLTFMaterial::diffuse_transmission_texture_offset,
                &GLTFMaterial::set_diffuse_transmission_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_texture_rotation",
                &GLTFMaterial::diffuse_transmission_texture_rotation,
                &GLTFMaterial::set_diffuse_transmission_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_texture_uv_scale",
                &GLTFMaterial::diffuse_transmission_texture_uv_scale,
                &GLTFMaterial::set_diffuse_transmission_texture_uv_scale,
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "diffuse_transmission_color_texture",
                &GLTFMaterial::m_diffuse_transmission_color_texture,
                "sRGB color texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "diffuse_transmission_color_texture_sampler",
                &GLTFMaterial::m_diffuse_transmission_color_texture_sampler,
                "sRGB color texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "diffuse_transmission_color_texture_path",
                &GLTFMaterial::m_diffuse_transmission_color_texture_path,
                "sRGB color texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_texcoord",
                &GLTFMaterial::diffuse_transmission_color_texture_texcoord,
                &GLTFMaterial::set_diffuse_transmission_color_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_offset",
                &GLTFMaterial::diffuse_transmission_color_texture_offset,
                &GLTFMaterial::set_diffuse_transmission_color_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_rotation",
                &GLTFMaterial::diffuse_transmission_color_texture_rotation,
                &GLTFMaterial::set_diffuse_transmission_color_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "diffuse_transmission_color_texture_uv_scale",
                &GLTFMaterial::diffuse_transmission_color_texture_uv_scale,
                &GLTFMaterial::set_diffuse_transmission_color_texture_uv_scale,
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "anisotropy_strength",
                &GLTFMaterial::m_anisotropy_strength,
                "Anisotropy strength; multiplied by texture blue.",
                reflection::default_value(0.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_rw(
                "anisotropy_rotation",
                &GLTFMaterial::m_anisotropy_rotation,
                "Anisotropy direction rotation in radians.",
                reflection::default_value(0.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_rw(
                "diffuse_transmission_factor",
                &GLTFMaterial::m_diffuse_transmission_factor,
                "Fraction of remaining dielectric diffuse energy transmitted; texture alpha.",
                reflection::default_value(0.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_rw(
                "diffuse_transmission_color_factor",
                &GLTFMaterial::m_diffuse_transmission_color_factor,
                "Linear diffuse transmission color; multiplied by sRGB texture RGB.",
                reflection::default_value(float3(1.f)),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_rw(
                "sheen_color_texture",
                &GLTFMaterial::m_sheen_color_texture,
                "sRGB sheen color texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "sheen_color_texture_sampler",
                &GLTFMaterial::m_sheen_color_texture_sampler,
                "sRGB sheen color texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "sheen_color_texture_path",
                &GLTFMaterial::m_sheen_color_texture_path,
                "sRGB sheen color texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "sheen_color_texture_texcoord",
                &GLTFMaterial::sheen_color_texture_texcoord,
                &GLTFMaterial::set_sheen_color_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_color_texture_offset",
                &GLTFMaterial::sheen_color_texture_offset,
                &GLTFMaterial::set_sheen_color_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_color_texture_rotation",
                &GLTFMaterial::sheen_color_texture_rotation,
                &GLTFMaterial::set_sheen_color_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_color_texture_uv_scale",
                &GLTFMaterial::sheen_color_texture_uv_scale,
                &GLTFMaterial::set_sheen_color_texture_uv_scale,
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "sheen_roughness_texture",
                &GLTFMaterial::m_sheen_roughness_texture,
                "Linear alpha sheen roughness texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "sheen_roughness_texture_sampler",
                &GLTFMaterial::m_sheen_roughness_texture_sampler,
                "Linear alpha sheen roughness texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "sheen_roughness_texture_path",
                &GLTFMaterial::m_sheen_roughness_texture_path,
                "Linear alpha sheen roughness texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "sheen_roughness_texture_texcoord",
                &GLTFMaterial::sheen_roughness_texture_texcoord,
                &GLTFMaterial::set_sheen_roughness_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_roughness_texture_offset",
                &GLTFMaterial::sheen_roughness_texture_offset,
                &GLTFMaterial::set_sheen_roughness_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_roughness_texture_rotation",
                &GLTFMaterial::sheen_roughness_texture_rotation,
                &GLTFMaterial::set_sheen_roughness_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "sheen_roughness_texture_uv_scale",
                &GLTFMaterial::sheen_roughness_texture_uv_scale,
                &GLTFMaterial::set_sheen_roughness_texture_uv_scale,
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "iridescence_texture",
                &GLTFMaterial::m_iridescence_texture,
                "Linear red iridescence strength texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "iridescence_texture_sampler",
                &GLTFMaterial::m_iridescence_texture_sampler,
                "Linear red iridescence strength texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "iridescence_texture_path",
                &GLTFMaterial::m_iridescence_texture_path,
                "Linear red iridescence strength texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "iridescence_texture_texcoord",
                &GLTFMaterial::iridescence_texture_texcoord,
                &GLTFMaterial::set_iridescence_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_texture_offset",
                &GLTFMaterial::iridescence_texture_offset,
                &GLTFMaterial::set_iridescence_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_texture_rotation",
                &GLTFMaterial::iridescence_texture_rotation,
                &GLTFMaterial::set_iridescence_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_texture_uv_scale",
                &GLTFMaterial::iridescence_texture_uv_scale,
                &GLTFMaterial::set_iridescence_texture_uv_scale,
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "iridescence_thickness_texture",
                &GLTFMaterial::m_iridescence_thickness_texture,
                "Linear green iridescence thickness texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "iridescence_thickness_texture_sampler",
                &GLTFMaterial::m_iridescence_thickness_texture_sampler,
                "Linear green iridescence thickness texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "iridescence_thickness_texture_path",
                &GLTFMaterial::m_iridescence_thickness_texture_path,
                "Linear green iridescence thickness texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "iridescence_thickness_texture_texcoord",
                &GLTFMaterial::iridescence_thickness_texture_texcoord,
                &GLTFMaterial::set_iridescence_thickness_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_thickness_texture_offset",
                &GLTFMaterial::iridescence_thickness_texture_offset,
                &GLTFMaterial::set_iridescence_thickness_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_thickness_texture_rotation",
                &GLTFMaterial::iridescence_thickness_texture_rotation,
                &GLTFMaterial::set_iridescence_thickness_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "iridescence_thickness_texture_uv_scale",
                &GLTFMaterial::iridescence_thickness_texture_uv_scale,
                &GLTFMaterial::set_iridescence_thickness_texture_uv_scale,
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "sheen_color_factor",
                &GLTFMaterial::m_sheen_color_factor,
                "Linear sheen reflectance; zero disables the layer.",
                reflection::default_value(float3(0.f)),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_rw(
                "sheen_roughness_factor",
                &GLTFMaterial::m_sheen_roughness_factor,
                "Perceptual sheen roughness, multiplied by texture alpha.",
                reflection::default_value(0.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_rw(
                "iridescence_factor",
                &GLTFMaterial::m_iridescence_factor,
                "Thin-film Fresnel weight, multiplied by texture red.",
                reflection::default_value(0.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_rw(
                "iridescence_ior",
                &GLTFMaterial::m_iridescence_ior,
                "Thin-film index of refraction (at least one).",
                reflection::default_value(1.3f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_rw(
                "iridescence_thickness_minimum",
                &GLTFMaterial::m_iridescence_thickness_minimum,
                "Thickness in nanometers at texture green zero; may exceed maximum.",
                reflection::default_value(100.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_rw(
                "iridescence_thickness_maximum",
                &GLTFMaterial::m_iridescence_thickness_maximum,
                "Thickness in nanometers at texture green one, or without a texture.",
                reflection::default_value(400.f),
                reflection::on_change(&GLTFMaterial::mark_dirty_extensions)
            )
            .def_rw(
                "clearcoat_texture",
                &GLTFMaterial::m_clearcoat_texture,
                "Linear red clearcoat texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "clearcoat_texture_sampler",
                &GLTFMaterial::m_clearcoat_texture_sampler,
                "Linear red clearcoat texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "clearcoat_texture_path",
                &GLTFMaterial::m_clearcoat_texture_path,
                "Linear red clearcoat texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_texture_texcoord",
                &GLTFMaterial::clearcoat_texture_texcoord,
                &GLTFMaterial::set_clearcoat_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_texture_offset",
                &GLTFMaterial::clearcoat_texture_offset,
                &GLTFMaterial::set_clearcoat_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_texture_rotation",
                &GLTFMaterial::clearcoat_texture_rotation,
                &GLTFMaterial::set_clearcoat_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_texture_uv_scale",
                &GLTFMaterial::clearcoat_texture_uv_scale,
                &GLTFMaterial::set_clearcoat_texture_uv_scale,
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "clearcoat_roughness_texture",
                &GLTFMaterial::m_clearcoat_roughness_texture,
                "Linear green clearcoat roughness texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "clearcoat_roughness_texture_sampler",
                &GLTFMaterial::m_clearcoat_roughness_texture_sampler,
                "Linear green clearcoat roughness texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "clearcoat_roughness_texture_path",
                &GLTFMaterial::m_clearcoat_roughness_texture_path,
                "Linear green clearcoat roughness texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_texcoord",
                &GLTFMaterial::clearcoat_roughness_texture_texcoord,
                &GLTFMaterial::set_clearcoat_roughness_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_offset",
                &GLTFMaterial::clearcoat_roughness_texture_offset,
                &GLTFMaterial::set_clearcoat_roughness_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_rotation",
                &GLTFMaterial::clearcoat_roughness_texture_rotation,
                &GLTFMaterial::set_clearcoat_roughness_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_roughness_texture_uv_scale",
                &GLTFMaterial::clearcoat_roughness_texture_uv_scale,
                &GLTFMaterial::set_clearcoat_roughness_texture_uv_scale,
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "clearcoat_normal_texture",
                &GLTFMaterial::m_clearcoat_normal_texture,
                "Tangent-space clearcoat normal texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "clearcoat_normal_texture_sampler",
                &GLTFMaterial::m_clearcoat_normal_texture_sampler,
                "Tangent-space clearcoat normal texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "clearcoat_normal_texture_path",
                &GLTFMaterial::m_clearcoat_normal_texture_path,
                "Tangent-space clearcoat normal texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "clearcoat_normal_texture_texcoord",
                &GLTFMaterial::clearcoat_normal_texture_texcoord,
                &GLTFMaterial::set_clearcoat_normal_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_normal_texture_offset",
                &GLTFMaterial::clearcoat_normal_texture_offset,
                &GLTFMaterial::set_clearcoat_normal_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_normal_texture_rotation",
                &GLTFMaterial::clearcoat_normal_texture_rotation,
                &GLTFMaterial::set_clearcoat_normal_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "clearcoat_normal_texture_uv_scale",
                &GLTFMaterial::clearcoat_normal_texture_uv_scale,
                &GLTFMaterial::set_clearcoat_normal_texture_uv_scale,
                "Authored glTF UV scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "clearcoat_factor",
                &GLTFMaterial::m_clearcoat_factor,
                "Clearcoat intensity, multiplied by linear texture red.",
                reflection::default_value(float16_t(0.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_clearcoat)
            )
            .def_rw(
                "clearcoat_roughness_factor",
                &GLTFMaterial::m_clearcoat_roughness_factor,
                "Clearcoat perceptual roughness, multiplied by linear texture green.",
                reflection::default_value(float16_t(0.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "clearcoat_normal_texture_scale",
                &GLTFMaterial::m_clearcoat_normal_texture_scale,
                "Scale of the clearcoat tangent-space normal X and Y components.",
                reflection::default_value(float16_t(1.f)),
                reflection::on_change(&GLTFMaterial::mark_dirty_clearcoat)
            )
            .def_rw(
                "transmission_texture",
                &GLTFMaterial::m_transmission_texture,
                "Linear red transmission texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "transmission_texture_sampler",
                &GLTFMaterial::m_transmission_texture_sampler,
                "Linear red transmission texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "transmission_texture_path",
                &GLTFMaterial::m_transmission_texture_path,
                "Linear red transmission texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "transmission_texture_texcoord",
                &GLTFMaterial::transmission_texture_texcoord,
                &GLTFMaterial::set_transmission_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "transmission_texture_offset",
                &GLTFMaterial::transmission_texture_offset,
                &GLTFMaterial::set_transmission_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "transmission_texture_rotation",
                &GLTFMaterial::transmission_texture_rotation,
                &GLTFMaterial::set_transmission_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "transmission_texture_uv_scale",
                &GLTFMaterial::transmission_texture_uv_scale,
                &GLTFMaterial::set_transmission_texture_uv_scale,
                "Authored glTF UV uv scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "specular_texture",
                &GLTFMaterial::m_specular_texture,
                "Linear alpha specular strength texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "specular_texture_sampler",
                &GLTFMaterial::m_specular_texture_sampler,
                "Linear alpha specular strength texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "specular_texture_path",
                &GLTFMaterial::m_specular_texture_path,
                "Linear alpha specular strength texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_texture_texcoord",
                &GLTFMaterial::specular_texture_texcoord,
                &GLTFMaterial::set_specular_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_texture_offset",
                &GLTFMaterial::specular_texture_offset,
                &GLTFMaterial::set_specular_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_texture_rotation",
                &GLTFMaterial::specular_texture_rotation,
                &GLTFMaterial::set_specular_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_texture_uv_scale",
                &GLTFMaterial::specular_texture_uv_scale,
                &GLTFMaterial::set_specular_texture_uv_scale,
                "Authored glTF UV uv scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "specular_color_texture",
                &GLTFMaterial::m_specular_color_texture,
                "sRGB specular color texture.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "specular_color_texture_sampler",
                &GLTFMaterial::m_specular_color_texture_sampler,
                "sRGB specular color texture sampler.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "specular_color_texture_path",
                &GLTFMaterial::m_specular_color_texture_path,
                "sRGB specular color texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "specular_color_texture_texcoord",
                &GLTFMaterial::specular_color_texture_texcoord,
                &GLTFMaterial::set_specular_color_texture_texcoord,
                "Texture-coordinate set.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_color_texture_offset",
                &GLTFMaterial::specular_color_texture_offset,
                &GLTFMaterial::set_specular_color_texture_offset,
                "Authored glTF UV offset.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_color_texture_rotation",
                &GLTFMaterial::specular_color_texture_rotation,
                &GLTFMaterial::set_specular_color_texture_rotation,
                "Authored glTF UV rotation.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "specular_color_texture_uv_scale",
                &GLTFMaterial::specular_color_texture_uv_scale,
                &GLTFMaterial::set_specular_color_texture_uv_scale,
                "Authored glTF UV uv scale.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "specular_factor",
                &GLTFMaterial::m_specular_factor,
                "Dielectric specular strength, including grazing reflection; multiplied by texture alpha.",
                reflection::default_value(float16_t(1.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "specular_color_factor",
                &GLTFMaterial::m_specular_color_factor,
                "Linear dielectric F0 multiplier. Values above one are allowed; the resulting F0 is clamped.",
                reflection::default_value(float16_t3(1.f)),
                reflection::value_range_positive(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_prop_rw(
                "base_color_texture_offset",
                &GLTFMaterial::base_color_texture_offset,
                &GLTFMaterial::set_base_color_texture_offset,
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "base_color_texture_rotation",
                &GLTFMaterial::base_color_texture_rotation,
                &GLTFMaterial::set_base_color_texture_rotation,
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "base_color_texture_uv_scale",
                &GLTFMaterial::base_color_texture_uv_scale,
                &GLTFMaterial::set_base_color_texture_uv_scale,
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "metallic_roughness_texture_offset",
                &GLTFMaterial::metallic_roughness_texture_offset,
                &GLTFMaterial::set_metallic_roughness_texture_offset,
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "metallic_roughness_texture_rotation",
                &GLTFMaterial::metallic_roughness_texture_rotation,
                &GLTFMaterial::set_metallic_roughness_texture_rotation,
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "metallic_roughness_texture_uv_scale",
                &GLTFMaterial::metallic_roughness_texture_uv_scale,
                &GLTFMaterial::set_metallic_roughness_texture_uv_scale,
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "normal_texture_offset",
                &GLTFMaterial::normal_texture_offset,
                &GLTFMaterial::set_normal_texture_offset,
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "normal_texture_rotation",
                &GLTFMaterial::normal_texture_rotation,
                &GLTFMaterial::set_normal_texture_rotation,
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "normal_texture_uv_scale",
                &GLTFMaterial::normal_texture_uv_scale,
                &GLTFMaterial::set_normal_texture_uv_scale,
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "emissive_texture_offset",
                &GLTFMaterial::emissive_texture_offset,
                &GLTFMaterial::set_emissive_texture_offset,
                "UV offset in glTF upper-left coordinates.",
                reflection::default_value(float2(0.f)),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "emissive_texture_rotation",
                &GLTFMaterial::emissive_texture_rotation,
                &GLTFMaterial::set_emissive_texture_rotation,
                "UV rotation in radians in glTF upper-left coordinates.",
                reflection::default_value(0.f),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "emissive_texture_uv_scale",
                &GLTFMaterial::emissive_texture_uv_scale,
                &GLTFMaterial::set_emissive_texture_uv_scale,
                "UV scale applied before rotation and offset.",
                reflection::default_value(float2(1.f)),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "alpha_mode",
                &GLTFMaterial::m_alpha_mode,
                "Alpha handling mode.",
                reflection::default_value(AlphaMode::opaque),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "alpha_factor",
                &GLTFMaterial::m_alpha_factor,
                "Constant alpha factor.",
                reflection::default_value(1.f),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "alpha_cutoff",
                &GLTFMaterial::m_alpha_cutoff,
                "Alpha cutoff used for mask mode.",
                reflection::default_value(0.5f),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "use_vertex_color",
                &GLTFMaterial::m_use_vertex_color,
                "Multiply base color and opacity by COLOR_0 when available.",
                reflection::default_value(false),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "base_color_texture",
                &GLTFMaterial::m_base_color_texture,
                "Base color texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "base_color_texture_sampler",
                &GLTFMaterial::m_base_color_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "base_color_texture_path",
                &GLTFMaterial::m_base_color_texture_path,
                "Base color texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "base_color_texture_texcoord",
                &GLTFMaterial::base_color_texture_texcoord,
                &GLTFMaterial::set_base_color_texture_texcoord,
                "Texture-coordinate set used by the base color texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "base_color_factor",
                &GLTFMaterial::m_base_color_factor,
                "Base color factor",
                reflection::default_value(float16_t3{1.f, 1.f, 1.f}),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "metallic_roughness_texture",
                &GLTFMaterial::m_metallic_roughness_texture,
                "Metallic-roughness texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "metallic_roughness_texture_sampler",
                &GLTFMaterial::m_metallic_roughness_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "metallic_roughness_texture_path",
                &GLTFMaterial::m_metallic_roughness_texture_path,
                "Metallic-roughness texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "metallic_roughness_texture_texcoord",
                &GLTFMaterial::metallic_roughness_texture_texcoord,
                &GLTFMaterial::set_metallic_roughness_texture_texcoord,
                "Texture-coordinate set used by the metallic-roughness texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "metallic_factor",
                &GLTFMaterial::m_metallic_factor,
                "Metallic factor (blue texture channel)",
                reflection::default_value(float16_t(1.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "roughness_factor",
                &GLTFMaterial::m_roughness_factor,
                "Perceptual roughness factor (green texture channel)",
                reflection::default_value(float16_t(1.f)),
                reflection::value_range_unit(),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "normal_texture",
                &GLTFMaterial::m_normal_texture,
                "Normal texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "normal_texture_sampler",
                &GLTFMaterial::m_normal_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_rw(
                "normal_texture_path",
                &GLTFMaterial::m_normal_texture_path,
                "Normal texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "normal_texture_texcoord",
                &GLTFMaterial::normal_texture_texcoord,
                &GLTFMaterial::set_normal_texture_texcoord,
                "Texture-coordinate set used by the normal texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "normal_texture_scale",
                &GLTFMaterial::m_normal_texture_scale,
                "Normal texture scale",
                reflection::default_value(float16_t(1.f)),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )
            .def_rw(
                "emissive_texture",
                &GLTFMaterial::m_emissive_texture,
                "Emissive texture (takes precedence over path)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_emission_resources)
            )
            .def_rw(
                "emissive_texture_sampler",
                &GLTFMaterial::m_emissive_texture_sampler,
                "Texture sampler (null uses the scene default)",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_emission_resources)
            )
            .def_rw(
                "emissive_texture_path",
                &GLTFMaterial::m_emissive_texture_path,
                "Emissive texture path",
                reflection::UIFlags::advanced,
                reflection::on_change(&GLTFMaterial::mark_dirty_emission_resources)
            )
            .def_prop_rw(
                "emissive_texture_texcoord",
                &GLTFMaterial::emissive_texture_texcoord,
                &GLTFMaterial::set_emissive_texture_texcoord,
                "Texture-coordinate set used by the emissive texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "emissive_factor",
                &GLTFMaterial::m_emissive_factor,
                "Emissive factor",
                reflection::default_value(float3(0.f)),
                reflection::value_range_positive(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&GLTFMaterial::mark_dirty_emission_properties)
            )

            .def_rw(
                "ior",
                &GLTFMaterial::m_ior,
                "Dielectric index of refraction for Fresnel and thin-transmission roughness",
                reflection::default_value(float16_t(1.5f)),
                reflection::value_range(0.0, 10.0),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            )

            .def_rw(
                "double_sided",
                &GLTFMaterial::m_double_sided,
                "Double-sided material",
                reflection::default_value(false),
                reflection::on_change(&GLTFMaterial::mark_dirty_properties)
            );
    }

private:
    uint32_t anisotropy_texture_texcoord() const { return m_anisotropy_texture_texcoord; }
    void set_anisotropy_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_anisotropy_texture_texcoord, value, "anisotropy_texture_texcoord");
        m_transforms_dirty = true;
    }
    float2 anisotropy_texture_offset() const { return m_texture_transforms[14].offset; }
    void set_anisotropy_texture_offset(float2 value)
    {
        m_texture_transforms[14].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float anisotropy_texture_rotation() const { return m_texture_transforms[14].rotation; }
    void set_anisotropy_texture_rotation(float value)
    {
        m_texture_transforms[14].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 anisotropy_texture_uv_scale() const { return m_texture_transforms[14].scale; }
    void set_anisotropy_texture_uv_scale(float2 value)
    {
        m_texture_transforms[14].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t diffuse_transmission_texture_texcoord() const { return m_diffuse_transmission_texture_texcoord; }
    void set_diffuse_transmission_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_diffuse_transmission_texture_texcoord, value, "diffuse_transmission_texture_texcoord");
        m_transforms_dirty = true;
    }
    float2 diffuse_transmission_texture_offset() const { return m_texture_transforms[15].offset; }
    void set_diffuse_transmission_texture_offset(float2 value)
    {
        m_texture_transforms[15].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float diffuse_transmission_texture_rotation() const { return m_texture_transforms[15].rotation; }
    void set_diffuse_transmission_texture_rotation(float value)
    {
        m_texture_transforms[15].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 diffuse_transmission_texture_uv_scale() const { return m_texture_transforms[15].scale; }
    void set_diffuse_transmission_texture_uv_scale(float2 value)
    {
        m_texture_transforms[15].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t diffuse_transmission_color_texture_texcoord() const
    {
        return m_diffuse_transmission_color_texture_texcoord;
    }
    void set_diffuse_transmission_color_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(
            m_diffuse_transmission_color_texture_texcoord,
            value,
            "diffuse_transmission_color_texture_texcoord"
        );
        m_transforms_dirty = true;
    }
    float2 diffuse_transmission_color_texture_offset() const { return m_texture_transforms[16].offset; }
    void set_diffuse_transmission_color_texture_offset(float2 value)
    {
        m_texture_transforms[16].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float diffuse_transmission_color_texture_rotation() const { return m_texture_transforms[16].rotation; }
    void set_diffuse_transmission_color_texture_rotation(float value)
    {
        m_texture_transforms[16].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 diffuse_transmission_color_texture_uv_scale() const { return m_texture_transforms[16].scale; }
    void set_diffuse_transmission_color_texture_uv_scale(float2 value)
    {
        m_texture_transforms[16].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t sheen_color_texture_texcoord() const { return m_sheen_color_texture_texcoord; }
    void set_sheen_color_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_sheen_color_texture_texcoord, value, "sheen_color_texture_texcoord");
        m_transforms_dirty = true;
    }
    float2 sheen_color_texture_offset() const { return m_texture_transforms[10].offset; }
    void set_sheen_color_texture_offset(float2 value)
    {
        m_texture_transforms[10].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float sheen_color_texture_rotation() const { return m_texture_transforms[10].rotation; }
    void set_sheen_color_texture_rotation(float value)
    {
        m_texture_transforms[10].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 sheen_color_texture_uv_scale() const { return m_texture_transforms[10].scale; }
    void set_sheen_color_texture_uv_scale(float2 value)
    {
        m_texture_transforms[10].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t sheen_roughness_texture_texcoord() const { return m_sheen_roughness_texture_texcoord; }
    void set_sheen_roughness_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_sheen_roughness_texture_texcoord, value, "sheen_roughness_texture_texcoord");
        m_transforms_dirty = true;
    }
    float2 sheen_roughness_texture_offset() const { return m_texture_transforms[11].offset; }
    void set_sheen_roughness_texture_offset(float2 value)
    {
        m_texture_transforms[11].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float sheen_roughness_texture_rotation() const { return m_texture_transforms[11].rotation; }
    void set_sheen_roughness_texture_rotation(float value)
    {
        m_texture_transforms[11].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 sheen_roughness_texture_uv_scale() const { return m_texture_transforms[11].scale; }
    void set_sheen_roughness_texture_uv_scale(float2 value)
    {
        m_texture_transforms[11].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t iridescence_texture_texcoord() const { return m_iridescence_texture_texcoord; }
    void set_iridescence_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_iridescence_texture_texcoord, value, "iridescence_texture_texcoord");
        m_transforms_dirty = true;
    }
    float2 iridescence_texture_offset() const { return m_texture_transforms[12].offset; }
    void set_iridescence_texture_offset(float2 value)
    {
        m_texture_transforms[12].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float iridescence_texture_rotation() const { return m_texture_transforms[12].rotation; }
    void set_iridescence_texture_rotation(float value)
    {
        m_texture_transforms[12].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 iridescence_texture_uv_scale() const { return m_texture_transforms[12].scale; }
    void set_iridescence_texture_uv_scale(float2 value)
    {
        m_texture_transforms[12].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t iridescence_thickness_texture_texcoord() const { return m_iridescence_thickness_texture_texcoord; }
    void set_iridescence_thickness_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_iridescence_thickness_texture_texcoord, value, "iridescence_thickness_texture_texcoord");
        m_transforms_dirty = true;
    }
    float2 iridescence_thickness_texture_offset() const { return m_texture_transforms[13].offset; }
    void set_iridescence_thickness_texture_offset(float2 value)
    {
        m_texture_transforms[13].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float iridescence_thickness_texture_rotation() const { return m_texture_transforms[13].rotation; }
    void set_iridescence_thickness_texture_rotation(float value)
    {
        m_texture_transforms[13].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 iridescence_thickness_texture_uv_scale() const { return m_texture_transforms[13].scale; }
    void set_iridescence_thickness_texture_uv_scale(float2 value)
    {
        m_texture_transforms[13].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t clearcoat_texture_texcoord() const { return m_clearcoat_texture_texcoord; }
    void set_clearcoat_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_clearcoat_texture_texcoord, value, "clearcoat_texture_texcoord");
        m_transforms_dirty = true;
    }
    float2 clearcoat_texture_offset() const { return m_texture_transforms[7].offset; }
    void set_clearcoat_texture_offset(float2 value)
    {
        m_texture_transforms[7].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float clearcoat_texture_rotation() const { return m_texture_transforms[7].rotation; }
    void set_clearcoat_texture_rotation(float value)
    {
        m_texture_transforms[7].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 clearcoat_texture_uv_scale() const { return m_texture_transforms[7].scale; }
    void set_clearcoat_texture_uv_scale(float2 value)
    {
        m_texture_transforms[7].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t clearcoat_roughness_texture_texcoord() const { return m_clearcoat_roughness_texture_texcoord; }
    void set_clearcoat_roughness_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_clearcoat_roughness_texture_texcoord, value, "clearcoat_roughness_texture_texcoord");
        m_transforms_dirty = true;
    }
    float2 clearcoat_roughness_texture_offset() const { return m_texture_transforms[8].offset; }
    void set_clearcoat_roughness_texture_offset(float2 value)
    {
        m_texture_transforms[8].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float clearcoat_roughness_texture_rotation() const { return m_texture_transforms[8].rotation; }
    void set_clearcoat_roughness_texture_rotation(float value)
    {
        m_texture_transforms[8].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 clearcoat_roughness_texture_uv_scale() const { return m_texture_transforms[8].scale; }
    void set_clearcoat_roughness_texture_uv_scale(float2 value)
    {
        m_texture_transforms[8].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t clearcoat_normal_texture_texcoord() const { return m_clearcoat_normal_texture_texcoord; }
    void set_clearcoat_normal_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_clearcoat_normal_texture_texcoord, value, "clearcoat_normal_texture_texcoord");
        m_transforms_dirty = true;
    }
    float2 clearcoat_normal_texture_offset() const { return m_texture_transforms[9].offset; }
    void set_clearcoat_normal_texture_offset(float2 value)
    {
        m_texture_transforms[9].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float clearcoat_normal_texture_rotation() const { return m_texture_transforms[9].rotation; }
    void set_clearcoat_normal_texture_rotation(float value)
    {
        m_texture_transforms[9].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 clearcoat_normal_texture_uv_scale() const { return m_texture_transforms[9].scale; }
    void set_clearcoat_normal_texture_uv_scale(float2 value)
    {
        m_texture_transforms[9].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t transmission_texture_texcoord() const { return m_transmission_texture_texcoord; }
    void set_transmission_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_transmission_texture_texcoord, value, "transmission_texture_texcoord");
    }
    float2 transmission_texture_offset() const { return m_texture_transforms[6].offset; }
    void set_transmission_texture_offset(float2 value)
    {
        m_texture_transforms[6].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float transmission_texture_rotation() const { return m_texture_transforms[6].rotation; }
    void set_transmission_texture_rotation(float value)
    {
        m_texture_transforms[6].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 transmission_texture_uv_scale() const { return m_texture_transforms[6].scale; }
    void set_transmission_texture_uv_scale(float2 value)
    {
        m_texture_transforms[6].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t specular_texture_texcoord() const { return m_specular_texture_texcoord; }
    void set_specular_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_specular_texture_texcoord, value, "specular_texture_texcoord");
    }
    float2 specular_texture_offset() const { return m_texture_transforms[4].offset; }
    void set_specular_texture_offset(float2 value)
    {
        m_texture_transforms[4].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float specular_texture_rotation() const { return m_texture_transforms[4].rotation; }
    void set_specular_texture_rotation(float value)
    {
        m_texture_transforms[4].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 specular_texture_uv_scale() const { return m_texture_transforms[4].scale; }
    void set_specular_texture_uv_scale(float2 value)
    {
        m_texture_transforms[4].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    uint32_t specular_color_texture_texcoord() const { return m_specular_color_texture_texcoord; }
    void set_specular_color_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_specular_color_texture_texcoord, value, "specular_color_texture_texcoord");
    }
    float2 specular_color_texture_offset() const { return m_texture_transforms[5].offset; }
    void set_specular_color_texture_offset(float2 value)
    {
        m_texture_transforms[5].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float specular_color_texture_rotation() const { return m_texture_transforms[5].rotation; }
    void set_specular_color_texture_rotation(float value)
    {
        m_texture_transforms[5].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 specular_color_texture_uv_scale() const { return m_texture_transforms[5].scale; }
    void set_specular_color_texture_uv_scale(float2 value)
    {
        m_texture_transforms[5].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    float2 base_color_texture_offset() const { return m_texture_transforms[0].offset; }
    void set_base_color_texture_offset(float2 value)
    {
        m_texture_transforms[0].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }

    float base_color_texture_rotation() const { return m_texture_transforms[0].rotation; }
    void set_base_color_texture_rotation(float value)
    {
        m_texture_transforms[0].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }

    float2 base_color_texture_uv_scale() const { return m_texture_transforms[0].scale; }
    void set_base_color_texture_uv_scale(float2 value)
    {
        m_texture_transforms[0].scale = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }

    float2 metallic_roughness_texture_offset() const { return m_texture_transforms[1].offset; }
    void set_metallic_roughness_texture_offset(float2 value)
    {
        m_texture_transforms[1].offset = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }

    float metallic_roughness_texture_rotation() const { return m_texture_transforms[1].rotation; }
    void set_metallic_roughness_texture_rotation(float value)
    {
        m_texture_transforms[1].rotation = value;
        m_transforms_dirty = true;
        mark_dirty_properties();
    }

    float2 metallic_roughness_texture_uv_scale() const { return m_texture_transforms[1].scale; }
    void set_metallic_roughness_texture_uv_scale(float2 value)
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
    void mark_dirty_clearcoat()
    {
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    void mark_dirty_extensions()
    {
        m_transforms_dirty = true;
        mark_dirty_properties();
    }
    void mark_dirty_volume()
    {
        // A live thin-to-solid edit must retain the exit boundary in traversal.
        if (m_thickness_factor > 0.f)
            set_enable_backface_culling(false);
        m_transforms_dirty = true;
        mark_dirty_properties();
    }

    void set_texture_texcoord(uint32_t& destination, uint32_t value, const char* property_name)
    {
        FALCOR_CHECK(value < 8, "{} must be in the range [0, 7], got {}.", property_name, value);
        if (destination != value) {
            destination = value;
            mark_dirty_properties();
        }
    }

    uint32_t base_color_texture_texcoord() const { return m_base_color_texture_texcoord; }
    void set_base_color_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_base_color_texture_texcoord, value, "base_color_texture_texcoord");
    }
    uint32_t metallic_roughness_texture_texcoord() const { return m_metallic_roughness_texture_texcoord; }
    void set_metallic_roughness_texture_texcoord(uint32_t value)
    {
        set_texture_texcoord(m_metallic_roughness_texture_texcoord, value, "metallic_roughness_texture_texcoord");
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

    ref<sgl::Texture> m_anisotropy_texture;
    ref<sgl::Sampler> m_anisotropy_texture_sampler;
    std::filesystem::path m_anisotropy_texture_path;
    uint32_t m_anisotropy_texture_texcoord{0};
    TextureHandle m_anisotropy_texture_handle;
    ref<sgl::Texture> m_diffuse_transmission_texture;
    ref<sgl::Sampler> m_diffuse_transmission_texture_sampler;
    std::filesystem::path m_diffuse_transmission_texture_path;
    uint32_t m_diffuse_transmission_texture_texcoord{0};
    TextureHandle m_diffuse_transmission_texture_handle;
    ref<sgl::Texture> m_diffuse_transmission_color_texture;
    ref<sgl::Sampler> m_diffuse_transmission_color_texture_sampler;
    std::filesystem::path m_diffuse_transmission_color_texture_path;
    uint32_t m_diffuse_transmission_color_texture_texcoord{0};
    TextureHandle m_diffuse_transmission_color_texture_handle;
    float m_anisotropy_strength{0.f};
    float m_anisotropy_rotation{0.f};
    float m_diffuse_transmission_factor{0.f};
    float3 m_diffuse_transmission_color_factor{float3(1.f)};
    bool m_has_surface_extensions{false};
    ref<sgl::Texture> m_sheen_color_texture;
    ref<sgl::Sampler> m_sheen_color_texture_sampler;
    std::filesystem::path m_sheen_color_texture_path;
    uint32_t m_sheen_color_texture_texcoord{0};
    TextureHandle m_sheen_color_texture_handle;
    ref<sgl::Texture> m_sheen_roughness_texture;
    ref<sgl::Sampler> m_sheen_roughness_texture_sampler;
    std::filesystem::path m_sheen_roughness_texture_path;
    uint32_t m_sheen_roughness_texture_texcoord{0};
    TextureHandle m_sheen_roughness_texture_handle;
    ref<sgl::Texture> m_iridescence_texture;
    ref<sgl::Sampler> m_iridescence_texture_sampler;
    std::filesystem::path m_iridescence_texture_path;
    uint32_t m_iridescence_texture_texcoord{0};
    TextureHandle m_iridescence_texture_handle;
    ref<sgl::Texture> m_iridescence_thickness_texture;
    ref<sgl::Sampler> m_iridescence_thickness_texture_sampler;
    std::filesystem::path m_iridescence_thickness_texture_path;
    uint32_t m_iridescence_thickness_texture_texcoord{0};
    TextureHandle m_iridescence_thickness_texture_handle;
    float3 m_sheen_color_factor{float3(0.f)};
    float m_sheen_roughness_factor{0.f};
    float m_iridescence_factor{0.f};
    float m_iridescence_ior{1.3f};
    float m_iridescence_thickness_minimum{100.f};
    float m_iridescence_thickness_maximum{400.f};
    uint32_t m_sheen_offset{0xffffffffu};
    uint32_t m_iridescence_offset{0xffffffffu};
    ref<sgl::Texture> m_clearcoat_texture;
    ref<sgl::Sampler> m_clearcoat_texture_sampler;
    std::filesystem::path m_clearcoat_texture_path;
    uint32_t m_clearcoat_texture_texcoord{0};
    TextureHandle m_clearcoat_texture_handle;
    ref<sgl::Texture> m_clearcoat_roughness_texture;
    ref<sgl::Sampler> m_clearcoat_roughness_texture_sampler;
    std::filesystem::path m_clearcoat_roughness_texture_path;
    uint32_t m_clearcoat_roughness_texture_texcoord{0};
    TextureHandle m_clearcoat_roughness_texture_handle;
    ref<sgl::Texture> m_clearcoat_normal_texture;
    ref<sgl::Sampler> m_clearcoat_normal_texture_sampler;
    std::filesystem::path m_clearcoat_normal_texture_path;
    uint32_t m_clearcoat_normal_texture_texcoord{0};
    TextureHandle m_clearcoat_normal_texture_handle;
    float16_t m_clearcoat_factor{0.f};
    float16_t m_clearcoat_roughness_factor{0.f};
    float16_t m_clearcoat_normal_texture_scale{1.f};
    uint32_t m_clearcoat_offset{0};
    ref<sgl::Texture> m_transmission_texture;
    ref<sgl::Sampler> m_transmission_texture_sampler;
    std::filesystem::path m_transmission_texture_path;
    uint32_t m_transmission_texture_texcoord{0};
    TextureHandle m_transmission_texture_handle;
    float16_t m_transmission_factor{0.f};
    ref<sgl::Texture> m_specular_texture;
    ref<sgl::Sampler> m_specular_texture_sampler;
    std::filesystem::path m_specular_texture_path;
    uint32_t m_specular_texture_texcoord{0};
    TextureHandle m_specular_texture_handle;
    ref<sgl::Texture> m_specular_color_texture;
    ref<sgl::Sampler> m_specular_color_texture_sampler;
    std::filesystem::path m_specular_color_texture_path;
    uint32_t m_specular_color_texture_texcoord{0};
    TextureHandle m_specular_color_texture_handle;
    float16_t m_specular_factor{1.f};
    float16_t3 m_specular_color_factor{1.f};
    GLTFTextureTransform m_texture_transforms[17];
    uint32_t m_transform_mask{0};
    uint32_t m_transform_offset{0};
    float m_thickness_factor{0.f};
    float3 m_attenuation_color{1.f};
    float m_attenuation_distance{std::numeric_limits<float>::infinity()};
    bool m_transforms_dirty{true};
    ref<SceneGlobals> m_scattering_globals;

    // Static properties.
    AlphaMode m_alpha_mode{AlphaMode::opaque};
    float m_alpha_factor{1.f};
    float m_alpha_cutoff{0.5f};
    bool m_use_vertex_color{false};
    ref<sgl::Texture> m_base_color_texture;
    ref<sgl::Sampler> m_base_color_texture_sampler;
    std::filesystem::path m_base_color_texture_path;
    float16_t3 m_base_color_factor{1.f, 1.f, 1.f};
    ref<sgl::Texture> m_metallic_roughness_texture;
    ref<sgl::Sampler> m_metallic_roughness_texture_sampler;
    std::filesystem::path m_metallic_roughness_texture_path;
    float16_t m_metallic_factor{1.f};
    float16_t m_roughness_factor{1.f};
    ref<sgl::Texture> m_normal_texture;
    ref<sgl::Sampler> m_normal_texture_sampler;
    std::filesystem::path m_normal_texture_path;
    float16_t m_normal_texture_scale{1.f};
    ref<sgl::Texture> m_emissive_texture;
    ref<sgl::Sampler> m_emissive_texture_sampler;
    std::filesystem::path m_emissive_texture_path;
    float3 m_emissive_factor{0.f, 0.f, 0.f};
    float16_t m_ior{1.5f};
    bool m_double_sided{false};
    uint32_t m_base_color_texture_texcoord{0};
    uint32_t m_metallic_roughness_texture_texcoord{0};
    uint32_t m_normal_texture_texcoord{0};
    uint32_t m_emissive_texture_texcoord{0};

    TextureHandle m_base_color_texture_handle;
    TextureHandle m_normal_texture_handle;
    TextureHandle m_metallic_roughness_texture_handle;
    TextureHandle m_emissive_texture_handle;
};

} // namespace falcor
