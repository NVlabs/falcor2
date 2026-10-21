// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gltf_conversions.h"
#include "gltf_conversion_utils.h"
#include "falcor2/render/scene_types.h"
#include <cmath>
#include <limits>

namespace falcor {

std::optional<Properties> gltf_to_standard(
    const ImporterMaterial& source,
    std::span<const ImporterTexture> textures,
    bool use_vertex_color,
    const GLTFTextureSetter& set_texture
)
{
    const auto& p = source.params;
    const auto type = p.get<std::string>("_type", "");
    if (type == "gltf_unlit")
        return gltf_to_gltf(source, textures, use_vertex_color, set_texture);
    if (type != "gltf_pbrMetallicRoughness" && type != "gltf_pbrSpecularGlossiness")
        return {};
    const bool sg = type == "gltf_pbrSpecularGlossiness";
    Properties r;
    r.set("_scene_material_type", sg ? "StandardSpecGlossMaterial" : "StandardMaterial");
    r.set("alpha_mode", p.get("alphaMode", AlphaMode::opaque));
    r.set("alpha_cutoff", p.get("alphaCutoff", 0.5f));
    r.set("use_vertex_color", use_vertex_color);
    r.set("double_sided", true);
    r.set("enable_backface_culling", !p.get("doubleSided", false) && p.get("thicknessFactor", 0.f) == 0.f);
    const float4 base = p.get(sg ? "diffuseFactor" : "baseColorFactor", float4(1.f));
    r.set(sg ? "diffuse_factor" : "base_color_factor", base.xyz());
    r.set("alpha_factor", base.w);
    auto texture = [&](std::string_view input, std::string_view output, TextureLoadFlags flags)
    {
        gltf_conversion::texture(source, r, textures, set_texture, input, output, flags, "Standard");
    };
    texture(
        sg ? "diffuseTexture" : "baseColorTexture",
        sg ? "diffuse_texture" : "base_color_texture",
        TextureLoadFlags::color
    );
    texture("normalTexture", "normal_texture", TextureLoadFlags::normal);
    r.set("normal_texture_scale", p.get("normalTexture:strength", 1.f));
    r.set("emissive_factor", p.get("emissiveFactor", float3(0.f)));
    texture("emissiveTexture", "emissive_texture", TextureLoadFlags::color);
    if (sg) {
        r.set("specular_factor", p.get("specularFactor", float3(1.f)));
        r.set("glossiness_factor", p.get("glossinessFactor", 1.f));
        texture("specularGlossinessTexture", "specular_glossiness_texture", TextureLoadFlags::color);
    } else {
        r.set("metallic_factor", p.get("metallicFactor", 1.f));
        r.set("roughness_factor", p.get("roughnessFactor", 1.f));
        r.set("metallic_texture_channel", 2u);
        r.set("roughness_texture_channel", 1u);
        texture("metallicRoughnessTexture", "metallic_roughness_texture", TextureLoadFlags::data_rgb);
        r.set("ior", p.get("ior", 1.5f));
        r.set("thin_walled", p.get("thicknessFactor", 0.f) == 0.f);
        r.set("specular_transmission_factor", p.get("transmissionFactor", 0.f));
        r.set("diffuse_transmission_factor", p.get("diffuseTransmissionFactor", 0.f));
        const bool diffuse_only
            = p.get("transmissionFactor", 0.f) == 0.f && p.get("diffuseTransmissionFactor", 0.f) > 0.f;
        auto transmission_color = diffuse_only ? p.get("diffuseTransmissionColorFactor", float3(1.f)) : base.xyz();
        if (p.get("thicknessFactor", 0.f) > 0.f && p.get("transmissionFactor", 0.f) > 0.f) {
            if (!p.has_property("baseColorTexture") && !use_vertex_color
                && p.get("diffuseTransmissionFactor", 0.f) == 0.f) {
                // Standard takes the square root of solid transmission tint at each boundary.
                // glTF base color is already a per-boundary tint, independent of volume absorption.
                transmission_color *= transmission_color;
            } else {
                gltf_conversion::warn(
                    r,
                    source,
                    "Standard",
                    "Solid transmission boundary tint with textures, vertex colors or diffuse transmission is "
                    "approximate"
                );
            }
        }
        r.set("transmission_factor", transmission_color);
        texture(
            diffuse_only ? "diffuseTransmissionColorTexture" : "baseColorTexture",
            "transmission_texture",
            TextureLoadFlags::color
        );
        // Standard's transmission texture is an RGB tint, not a lobe weight.
        if (p.has_property("transmissionTexture") || p.has_property("diffuseTransmissionTexture"))
            gltf_conversion::warn(r, source, "Standard", "Transmission weight textures are unsupported; factors only");
        if (p.get("hasTransmission", false))
            gltf_conversion::warn(
                r,
                source,
                "Standard",
                "Transmission uses Standard's existing dielectric model; glTF thin/solid semantics differ"
            );
        if (!diffuse_only && p.get("diffuseTransmissionFactor", 0.f) > 0.f)
            gltf_conversion::warn(
                r,
                source,
                "Standard",
                "Specular and diffuse transmission share one tint; using base color"
            );
        const auto color = p.get("attenuationColor", float3(1.f));
        const float distance = p.get("attenuationDistance", std::numeric_limits<float>::infinity());
        float3 sigma(0.f);
        if (distance > 0.f && std::isfinite(distance))
            for (int i = 0; i < 3; ++i)
                sigma[i] = -std::log(std::max(color[i], 1e-6f)) / distance;
        r.set("volume_sigma_a", sigma);
        gltf_conversion::unsupported(
            source,
            r,
            "Standard",
            {"hasSpecular", "hasClearcoat", "hasSheen", "hasIridescence", "hasAnisotropy"}
        );
    }
    return r;
}

} // namespace falcor
