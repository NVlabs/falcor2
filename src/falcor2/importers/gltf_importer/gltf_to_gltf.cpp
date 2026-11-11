// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gltf_conversions.h"
#include "falcor2/core/error.h"
#include "falcor2/render/scene_types.h"

#include <limits>

namespace falcor {

std::optional<Properties> gltf_to_gltf(
    const ImporterMaterial& source,
    std::span<const ImporterTexture> textures,
    bool use_vertex_color,
    const std::function<void(Properties&, std::string_view, const ImporterTexture&, TextureLoadFlags)>& set_texture
)
{
    const Properties& params = source.params;
    const auto type = params.get_optional<std::string>("_type");
    if (!type
        || (*type != "gltf_unlit" && *type != "gltf_pbrMetallicRoughness" && *type != "gltf_pbrSpecularGlossiness"))
        return {};

    const bool unlit = *type == "gltf_unlit";
    const bool specgloss = *type == "gltf_pbrSpecularGlossiness";
    Properties properties;
    properties.set(
        "_scene_material_type",
        unlit           ? "UnlitMaterial"
            : specgloss ? "GLTFSpecGlossMaterial"
                        : "GLTFMaterial"
    );
    properties.set("alpha_mode", params.get("alphaMode", AlphaMode::opaque));
    properties.set("alpha_cutoff", params.get("alphaCutoff", 0.5f));
    properties.set("use_vertex_color", use_vertex_color);
    // Visibility follows glTF sidedness; surviving back faces still shade two-sided.
    properties.set("double_sided", true);
    properties.set(
        "enable_backface_culling",
        !params.get("doubleSided", false) && !params.get("hasTransmission", false)
    );

    auto texture = [&](std::string_view input, std::string_view output, TextureLoadFlags flags, bool transform = true)
    {
        const auto index = params.get_optional<int>(input);
        if (!index)
            return;
        FALCOR_CHECK(*index >= 0 && size_t(*index) < textures.size(), "Invalid glTF texture index for '{}'", input);
        const int uv = params.get(fmt::format("{}:texcoord", input), 0);
        FALCOR_CHECK(
            uv >= 0 && uv < 8,
            "glTF texture property '{}' selects TEXCOORD_{}, but Falcor2 supports TEXCOORD_0 through TEXCOORD_7.",
            input,
            uv
        );
        set_texture(properties, output, textures[*index], flags);
        properties.set(fmt::format("{}_texcoord", output), static_cast<uint32_t>(uv));
        if (transform) {
            properties.set(fmt::format("{}_offset", output), params.get(fmt::format("{}:offset", input), float2(0.f)));
            properties.set(fmt::format("{}_rotation", output), params.get(fmt::format("{}:rotation", input), 0.f));
            properties.set(fmt::format("{}_uv_scale", output), params.get(fmt::format("{}:scale", input), float2(1.f)));
        }
    };

    // Keep optional values optional: absent inputs retain the target's defaults.
    auto copy = [&]<typename T>(std::string_view input, std::string_view output)
    {
        if (auto value = params.get_optional<T>(input))
            properties.set(output, *value);
    };
    const std::string_view base_input = specgloss ? "diffuseFactor" : "baseColorFactor";
    if (auto base = params.get_optional<float4>(base_input)) {
        properties.set(specgloss ? "diffuse_factor" : "base_color_factor", base->xyz());
        properties.set("alpha_factor", base->w);
    }
    texture(
        specgloss ? "diffuseTexture" : "baseColorTexture",
        specgloss ? "diffuse_texture" : "base_color_texture",
        TextureLoadFlags::color,
        !unlit
    );
    if (unlit)
        return properties;

    texture("normalTexture", "normal_texture", TextureLoadFlags::normal);
    copy.operator()<float>("normalTexture:strength", "normal_texture_scale");
    copy.operator()<float3>("emissiveFactor", "emissive_factor");
    texture("emissiveTexture", "emissive_texture", TextureLoadFlags::color);

    if (specgloss) {
        copy.operator()<float3>("specularFactor", "specular_factor");
        copy.operator()<float>("glossinessFactor", "glossiness_factor");
        texture("specularGlossinessTexture", "specular_glossiness_texture", TextureLoadFlags::color);
        return properties;
    }

    // Thin transmission retains authored sidedness. Only volume boundaries
    // require both faces independently of doubleSided.
    const float thickness = params.get("thicknessFactor", 0.f);
    properties.set("enable_backface_culling", thickness == 0.f && !params.get("doubleSided", false));
    properties.set("thickness_factor", thickness);
    properties.set("attenuation_color", params.get("attenuationColor", float3(1.f)));
    properties.set("attenuation_distance", params.get("attenuationDistance", std::numeric_limits<float>::infinity()));

    copy.operator()<float>("metallicFactor", "metallic_factor");
    copy.operator()<float>("roughnessFactor", "roughness_factor");
    copy.operator()<float>("ior", "ior");
    texture("metallicRoughnessTexture", "metallic_roughness_texture", TextureLoadFlags::data_rgb);
    properties.set("specular_factor", params.get("specularFactor", 1.f));
    properties.set("specular_color_factor", params.get("specularColorFactor", float3(1.f)));
    texture("specularTexture", "specular_texture", TextureLoadFlags::data);
    texture("specularColorTexture", "specular_color_texture", TextureLoadFlags::color);
    properties.set("transmission_factor", params.get("transmissionFactor", 0.f));
    texture("transmissionTexture", "transmission_texture", TextureLoadFlags::data);

    properties.set("clearcoat_factor", params.get("clearcoatFactor", 0.f));
    properties.set("clearcoat_roughness_factor", params.get("clearcoatRoughnessFactor", 0.f));
    properties.set("clearcoat_normal_texture_scale", params.get("clearcoatNormalTexture:strength", 1.f));
    texture("clearcoatTexture", "clearcoat_texture", TextureLoadFlags::data);
    texture("clearcoatRoughnessTexture", "clearcoat_roughness_texture", TextureLoadFlags::data_rgb);
    texture("clearcoatNormalTexture", "clearcoat_normal_texture", TextureLoadFlags::normal);

    properties.set("sheen_color_factor", params.get("sheenColorFactor", float3(0.f)));
    properties.set("sheen_roughness_factor", params.get("sheenRoughnessFactor", 0.f));
    texture("sheenColorTexture", "sheen_color_texture", TextureLoadFlags::color);
    texture("sheenRoughnessTexture", "sheen_roughness_texture", TextureLoadFlags::data);
    properties.set("iridescence_factor", params.get("iridescenceFactor", 0.f));
    properties.set("iridescence_ior", params.get("iridescenceIor", 1.3f));
    properties.set("iridescence_thickness_minimum", params.get("iridescenceThicknessMinimum", 100.f));
    properties.set("iridescence_thickness_maximum", params.get("iridescenceThicknessMaximum", 400.f));
    texture("iridescenceTexture", "iridescence_texture", TextureLoadFlags::data);
    texture("iridescenceThicknessTexture", "iridescence_thickness_texture", TextureLoadFlags::data_rgb);

    properties.set("anisotropy_strength", params.get("anisotropyStrength", 0.f));
    properties.set("anisotropy_rotation", params.get("anisotropyRotation", 0.f));
    texture("anisotropyTexture", "anisotropy_texture", TextureLoadFlags::data_rgb);
    properties.set("diffuse_transmission_factor", params.get("diffuseTransmissionFactor", 0.f));
    properties.set("diffuse_transmission_color_factor", params.get("diffuseTransmissionColorFactor", float3(1.f)));
    texture("diffuseTransmissionTexture", "diffuse_transmission_texture", TextureLoadFlags::data);
    texture("diffuseTransmissionColorTexture", "diffuse_transmission_color_texture", TextureLoadFlags::color);
    return properties;
}

} // namespace falcor
