// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gltf_conversions.h"
#include "gltf_conversion_utils.h"
#include "falcor2/render/scene_types.h"
#include <cmath>

namespace falcor {

std::optional<Properties> gltf_to_openpbr(
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
    FALCOR_CHECK(type != "gltf_pbrSpecularGlossiness", "OpenPBR conversion does not support glTF SpecGloss");
    if (type != "gltf_pbrMetallicRoughness")
        return {};
    Properties r;
    r.set("_scene_material_type", "OpenPBRMaterial");
    const float thickness = p.get("thicknessFactor", 0.f);
    r.set("geometry_thin_walled", thickness == 0.f);
    r.set("use_vertex_color", use_vertex_color);
    r.set("enable_backface_culling", !p.get("doubleSided", false) && thickness == 0.f);
    const auto base = p.get("baseColorFactor", float4(1.f));
    r.set("base_color", base.xyz());
    r.set("base_metalness", p.get("metallicFactor", 1.f));
    r.set("specular_roughness", p.get("roughnessFactor", 1.f));
    const auto alpha = p.get("alphaMode", AlphaMode::opaque);
    if (alpha != AlphaMode::opaque)
        r.set("opacity_factor", base.w);
    if (alpha == AlphaMode::mask)
        r.set("opacity_threshold", p.get("alphaCutoff", 0.5f));
    auto texture = [&](std::string_view input, std::string_view output, TextureLoadFlags flags, int channel = -1)
    {
        gltf_conversion::texture(source, r, textures, set_texture, input, output, flags, "OpenPBR");
        if (channel >= 0)
            r.set(fmt::format("{}_channel", output), uint32_t(channel));
    };
    texture("baseColorTexture", "base_color_texture", TextureLoadFlags::color);
    if (alpha != AlphaMode::opaque)
        texture("baseColorTexture", "opacity_texture", TextureLoadFlags::data, 3);
    texture("metallicRoughnessTexture", "base_metalness_texture", TextureLoadFlags::data_rgb, 2);
    texture("metallicRoughnessTexture", "specular_roughness_texture", TextureLoadFlags::data_rgb, 1);
    texture("normalTexture", "normal_texture", TextureLoadFlags::normal);
    r.set("normal_texture_scale", p.get("normalTexture:strength", 1.f));
    r.set("emission_color", p.get("emissiveFactor", float3(0.f)));
    r.set("emission_luminance", 1.f);
    texture("emissiveTexture", "emission_color_texture", TextureLoadFlags::color);

    r.set("specular_ior", p.get("ior", 1.5f));
    r.set("specular_weight", p.get("specularFactor", 1.f));
    r.set("specular_color", p.get("specularColorFactor", float3(1.f)));
    texture("specularTexture", "specular_weight_texture", TextureLoadFlags::data, 3);
    texture("specularColorTexture", "specular_color_texture", TextureLoadFlags::color);
    r.set("coat_weight", p.get("clearcoatFactor", 0.f));
    r.set("coat_roughness", p.get("clearcoatRoughnessFactor", 0.f));
    r.set("coat_ior", 1.5f);
    texture("clearcoatTexture", "coat_weight_texture", TextureLoadFlags::data, 0);
    texture("clearcoatRoughnessTexture", "coat_roughness_texture", TextureLoadFlags::data_rgb, 1);
    if (p.has_property("clearcoatNormalTexture"))
        gltf_conversion::warn(r, source, "OpenPBR", "Separate clearcoat normal is unsupported");

    const auto sheen = p.get("sheenColorFactor", float3(0.f));
    r.set("fuzz_weight", p.get("hasSheen", false) ? 1.f : 0.f);
    r.set("fuzz_color", sheen);
    r.set("fuzz_roughness", p.get("sheenRoughnessFactor", 0.f));
    texture("sheenColorTexture", "fuzz_color_texture", TextureLoadFlags::color);
    texture("sheenRoughnessTexture", "fuzz_roughness_texture", TextureLoadFlags::data, 3);
    if (p.get("hasSheen", false))
        gltf_conversion::warn(r, source, "OpenPBR", "Sheen maps to OpenPBR fuzz; scattering models differ");
    r.set("transmission_weight", p.get("transmissionFactor", 0.f));
    texture("transmissionTexture", "transmission_weight_texture", TextureLoadFlags::data, 0);
    if (thickness > 0.f) {
        r.set("transmission_color", p.get("attenuationColor", float3(1.f)));
        gltf_conversion::copy<float>(p, r, "attenuationDistance", "transmission_depth");
    } else {
        r.set("transmission_color", base.xyz());
        texture("baseColorTexture", "transmission_color_texture", TextureLoadFlags::color);
    }
    const float anisotropy = p.get("anisotropyStrength", 0.f);
    r.set("specular_roughness_anisotropy", anisotropy);
    if (anisotropy > 0.f) {
        if (!p.has_property("metallicRoughnessTexture") && !p.has_property("anisotropyTexture")) {
            // Match glTF's two GGX widths using OpenPBR's different roughness parameterization.
            // Texture-dependent remapping would require expressions coupling both inputs.
            const float roughness = p.get("roughnessFactor", 1.f);
            const float alpha_b = roughness * roughness;
            const float alpha_t = alpha_b + (1.f - alpha_b) * anisotropy * anisotropy;
            r.set("specular_roughness", std::sqrt(std::sqrt((alpha_t * alpha_t + alpha_b * alpha_b) * 0.5f)));
            r.set("specular_roughness_anisotropy", alpha_t > 0.f ? 1.f - alpha_b / alpha_t : 0.f);
        } else {
            gltf_conversion::warn(r, source, "OpenPBR", "Textured anisotropy roughness mapping is approximate");
        }
        if (p.get("anisotropyRotation", 0.f) != 0.f)
            gltf_conversion::warn(r, source, "OpenPBR", "Anisotropy rotation is unsupported");
        if (p.has_property("anisotropyTexture"))
            gltf_conversion::warn(r, source, "OpenPBR", "Anisotropy direction and strength texture is unsupported");
    }
    r.set("thin_film_weight", p.get("iridescenceFactor", 0.f));
    r.set("thin_film_ior", p.get("iridescenceIor", 1.3f));
    // OpenPBR uses micrometres, glTF specifies nanometres.
    r.set("thin_film_thickness", p.get("iridescenceThicknessMaximum", 400.f) * 0.001f);
    texture("iridescenceTexture", "thin_film_weight_texture", TextureLoadFlags::data, 0);
    if (p.get("hasIridescence", false))
        gltf_conversion::warn(
            r,
            source,
            "OpenPBR",
            "Iridescence maps to OpenPBR thin film; thickness texture remapping is unsupported"
        );
    gltf_conversion::unsupported(source, r, "OpenPBR", {"hasDiffuseTransmission"});
    return r;
}

} // namespace falcor
