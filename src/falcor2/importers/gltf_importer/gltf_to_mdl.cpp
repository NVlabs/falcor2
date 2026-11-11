// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gltf_conversions.h"
#include "gltf_conversion_utils.h"
#include "falcor2/core/platform.h"
#include "falcor2/render/scene_types.h"
#include "falcor2/render/geometry/geometry_attribute_classification.h"

#include <sgl/core/crypto.h>
#include <fmt/ranges.h>

#include <limits>

namespace falcor {

std::optional<Properties> gltf_to_mdl(
    const ImporterMaterial& source,
    std::span<const ImporterTexture> textures,
    UVOrigin uv_origin,
    const std::function<void(Properties&, std::string_view, const ImporterTexture&, bool)>& set_texture
)
{
    const Properties& params = source.params;
    const auto type = params.get_optional<std::string>("_type");
    if (!type || (*type != "gltf_pbrMetallicRoughness" && *type != "gltf_pbrSpecularGlossiness"))
        return {};
    const bool specgloss = *type == "gltf_pbrSpecularGlossiness";
    const bool thin = !params.get("hasVolume", false) || params.get("thicknessFactor", 0.f) == 0.f;
    Properties properties;
    std::vector<std::string> declarations;
    std::vector<std::string> arguments;
    bool has_normal_texture = false;

    // Defaults belong to the shared definition, never to a particular model.
    // Class compilation retains these arguments even when their values are 0/1.
    auto parameter = [&]<typename T>(std::string name, std::string_view type, std::string_view default_value, T value)
    {
        declarations.push_back(fmt::format("    uniform {} {} = {}", type, name, default_value));
        properties.set(name, value);
        return name;
    };
    auto scalar = [&](std::string_view destination, std::string_view input, float fallback)
    {
        const std::string name(destination);
        parameter(name, "float", fmt::format("{:.6f}", fallback), params.get(input, fallback));
        arguments.push_back(fmt::format("    {0}: {0}", name));
    };
    auto color = [&](std::string_view destination, std::string_view input, float fallback)
    {
        const std::string name(destination);
        parameter(name, "color", fmt::format("color({:.6f})", fallback), params.get(input, float3(fallback)));
        arguments.push_back(fmt::format("    {0}: {0}", name));
    };

    auto bind_texture = [&](std::string_view destination, std::string_view input, bool srgb, bool normal = false)
    {
        const auto index = params.get_optional<int>(input);
        if (!index)
            return;
        FALCOR_CHECK(*index >= 0 && size_t(*index) < textures.size(), "Invalid glTF texture index for '{}'", input);
        const auto& texture = textures[*index];
        const std::string name(destination);
        const std::string map = name + "_map";
        declarations.push_back(fmt::format("    uniform texture_2d {} = texture_2d()", map));
        set_texture(properties, map, texture, srgb);

        const int uv = params.get(fmt::format("{}:texcoord", input), 0);
        FALCOR_CHECK(
            uv >= 0 && uv < 4,
            "MDL glTF material '{}' uses TEXCOORD_{}, but MDL supports TEXCOORD_0 through TEXCOORD_3",
            source.name,
            uv
        );
        const std::string uv_parameter = parameter(name + "_uv", "int", "0", uv);
        const auto wrap_value = [](TextureWrapMode mode)
        {
            switch (mode) {
            case TextureWrapMode::repeat:
                return 10497;
            case TextureWrapMode::mirror_repeat:
                return 33648;
            case TextureWrapMode::clamp_to_edge:
                return 33071;
            }
            FALCOR_UNREACHABLE();
        };
        const auto wrap_s = parameter(name + "_wrap_s", "gltf_wrapping_mode", "repeat", wrap_value(texture.wrap_s));
        const auto wrap_t = parameter(name + "_wrap_t", "gltf_wrapping_mode", "repeat", wrap_value(texture.wrap_t));
        std::string lookup = fmt::format(
            "    {}: {}(\n        texture: {},\n        tex_coord_index: {},\n        wrap_s: {},\n        wrap_t: {}",
            name,
            normal ? "falcor_gltf_normal_texture_lookup" : "gltf_texture_lookup",
            map,
            uv_parameter,
            wrap_s,
            wrap_t
        );
        if (normal) {
            has_normal_texture = true;
            const auto strength
                = parameter(name + "_scale", "float", "1.0", params.get(fmt::format("{}:strength", input), 1.f));
            const auto tangent_v_sign
                = parameter(name + "_tangent_v_sign", "float", "1.0", uv_origin == UVOrigin::upper_left ? -1.f : 1.f);
            lookup += fmt::format(
                ",\n        normal_scale_factor: {},\n        tex_tangent_index: {},\n        tangent_v_sign: {}",
                strength,
                uv_parameter,
                tangent_v_sign
            );
        }
        const bool transformed = params.get(fmt::format("{}:has_transform", input), false);
        if (transformed) {
            const auto offset = parameter(
                name + "_offset",
                "float2",
                "float2(0.0)",
                params.get(fmt::format("{}:offset", input), float2(0.f))
            );
            const auto rotation
                = parameter(name + "_rotation", "float", "0.0", params.get(fmt::format("{}:rotation", input), 0.f));
            const auto scale = parameter(
                name + "_uv_scale",
                "float2",
                "float2(1.0)",
                params.get(fmt::format("{}:scale", input), float2(1.f))
            );
            // The SDK colour helper already matches the Khronos renderer's
            // rotation in glTF's Y-down UVs; do not negate the angle. The GLSL
            // example in the extension README has the opposite sign (#1563).
            // Normal-frame fidelity remains a separately reported limitation.
            lookup += fmt::format(
                ",\n        offset: {},\n        rotation: {},\n        scale: {}",
                offset,
                rotation,
                scale
            );
        }
        lookup += "\n    )";
        if (input == "iridescenceThicknessTexture") {
            // The installed SDK consumes R for thickness, while glTF stores G.
            lookup.insert(lookup.find(": ") + 2, "gltf_green_to_red(");
            lookup += ")";
        }
        if (input == "anisotropyTexture") {
            // Reflect the encoded direction in the same V convention as the
            // rotation, without changing texture coordinates or strength (B).
            lookup.insert(lookup.find(": ") + 2, "falcor_gltf_anisotropy_texture(");
            lookup += ", anisotropy_tangent_v_sign)";
        }
        arguments.push_back(lookup);
        if (normal && (transformed || uv != 0))
            sgl::log_warn(
                "MDL glTF material '{}': normal texture '{}' uses secondary/transformed UVs; the MDL adapter currently "
                "supplies only the shared tangent frame",
                source.name,
                input
            );
    };

    const float4 base = params.get(specgloss ? "diffuseFactor" : "baseColorFactor", float4(1.f));
    const std::string base_name = specgloss ? "diffuse_factor" : "base_color_factor";
    parameter(base_name, "color", "color(1.0)", base.xyz());
    arguments.push_back(fmt::format("    {0}: {0}", base_name));
    bind_texture(
        specgloss ? "diffuse_texture" : "base_color_texture",
        specgloss ? "diffuseTexture" : "baseColorTexture",
        true
    );
    if (specgloss) {
        color("specular_factor", "specularFactor", 1.f);
        scalar("glossiness_factor", "glossinessFactor", 1.f);
        bind_texture("specular_glossiness_texture", "specularGlossinessTexture", true);
    } else {
        scalar("metallic_factor", "metallicFactor", 1.f);
        scalar("roughness_factor", "roughnessFactor", 1.f);
        scalar("ior", "ior", 1.5f);
        bind_texture("metallic_roughness_texture", "metallicRoughnessTexture", false);
        if (params.get("hasSpecular", false)) {
            scalar("specular_factor", "specularFactor", 1.f);
            color("specular_color_factor", "specularColorFactor", 1.f);
            bind_texture("specular_texture", "specularTexture", false);
            bind_texture("specular_color_texture", "specularColorTexture", true);
        }
        if (params.get("hasClearcoat", false)) {
            scalar("clearcoat_factor", "clearcoatFactor", 0.f);
            scalar("clearcoat_roughness_factor", "clearcoatRoughnessFactor", 0.f);
            bind_texture("clearcoat_texture", "clearcoatTexture", false);
            bind_texture("clearcoat_roughness_texture", "clearcoatRoughnessTexture", false);
            bind_texture("clearcoat_normal_texture", "clearcoatNormalTexture", false, true);
        }
        if (params.get("hasTransmission", false)) {
            scalar("transmission_factor", "transmissionFactor", 0.f);
            bind_texture("transmission_texture", "transmissionTexture", false);
        }
        if (params.get("hasSheen", false)) {
            color("sheen_color_factor", "sheenColorFactor", 0.f);
            scalar("sheen_roughness_factor", "sheenRoughnessFactor", 0.f);
            bind_texture("sheen_color_texture", "sheenColorTexture", true);
            bind_texture("sheen_roughness_texture", "sheenRoughnessTexture", false);
        }
        if (params.get("hasAnisotropy", false)) {
            scalar("anisotropy_strength", "anisotropyStrength", 0.f);
            parameter("anisotropy_rotation", "float", "0.0", params.get("anisotropyRotation", 0.f));
            parameter("anisotropy_tangent_v_sign", "float", "1.0", uv_origin == UVOrigin::upper_left ? -1.f : 1.f);
            // The MDL state converts the scene bitangent to lower-left V.
            // Keep the public glTF angle and compensate only the SDK argument.
            arguments.push_back("    anisotropy_rotation: anisotropy_rotation * anisotropy_tangent_v_sign");
            bind_texture("anisotropy_texture", "anisotropyTexture", false);
        }
        if (params.get("hasIridescence", false)) {
            scalar("iridescence_factor", "iridescenceFactor", 0.f);
            scalar("iridescence_ior", "iridescenceIor", 1.3f);
            scalar("iridescence_thickness_minimum", "iridescenceThicknessMinimum", 100.f);
            scalar("iridescence_thickness_maximum", "iridescenceThicknessMaximum", 400.f);
            bind_texture("iridescence_texture", "iridescenceTexture", false);
            bind_texture("iridescence_thickness_texture", "iridescenceThicknessTexture", false);
        }
        gltf_conversion::unsupported(source, properties, "MDL", {"hasDiffuseTransmission"});
        // The SDK traces actual boundary distances, not raster thickness maps.
        // Keep the boundary type constant because the renderer's material header
        // cannot yet represent a state-dependent MDL thin_walled expression.
        arguments.push_back(fmt::format("    thin_walled: {}", thin ? "true" : "false"));
        if (params.get("hasVolume", false)) {
            scalar("attenuation_distance", "attenuationDistance", std::numeric_limits<float>::max());
            color("attenuation_color", "attenuationColor", 1.f);
        }
    }
    bind_texture("normal_texture", "normalTexture", false, true);
    color("emissive_factor", "emissiveFactor", 0.f);
    bind_texture("emissive_texture", "emissiveTexture", true);

    const auto alpha = params.get("alphaMode", AlphaMode::opaque);
    arguments.push_back(
        fmt::format(
            "    alpha_mode: {}",
            alpha == AlphaMode::mask        ? "mask"
                : alpha == AlphaMode::blend ? "blend"
                                            : "opaque"
        )
    );
    if (alpha != AlphaMode::opaque) {
        parameter("base_alpha", "float", "1.0", base.w);
        arguments.push_back("    base_alpha: base_alpha");
    }
    if (alpha == AlphaMode::mask)
        scalar("alpha_cutoff", "alphaCutoff", 0.5f);

    // The SDK normal helper follows MDL's lower-left tangent convention and
    // blends toward the flat normal. glTF instead scales XY in its authored
    // tangent frame, so decode the linear lookup in the conversion itself.
    const std::string normal_helper = has_normal_texture ? R"mdl(
import ::math::*;
import ::state::*;

gltf_normal_texture_lookup_value falcor_gltf_normal_texture_lookup(
    uniform texture_2d texture,
    uniform int tex_coord_index = 0,
    uniform gltf_wrapping_mode wrap_s = repeat,
    uniform gltf_wrapping_mode wrap_t = repeat,
    uniform float normal_scale_factor = 1.0,
    uniform int tex_tangent_index = 0,
    uniform float tangent_v_sign = 1.0,
    uniform float2 offset = float2(0.0),
    uniform float rotation = 0.0,
    uniform float2 scale = float2(1.0)
) {
    gltf_texture_lookup_value sample = gltf_texture_lookup(
        texture: texture, tex_coord_index: tex_coord_index,
        wrap_s: wrap_s, wrap_t: wrap_t, offset: offset, rotation: rotation, scale: scale);
    float3 n = float3(sample.value.x, sample.value.y, sample.value.z) * 2.0 - 1.0;
    n = math::normalize(float3(n.x * normal_scale_factor, n.y * normal_scale_factor, n.z));
    // Undo the renderer's conversion of an upper-left scene tangent to MDL's V.
    float3 normal = math::normalize(
        n.x * state::texture_tangent_u(tex_tangent_index)
        + n.y * tangent_v_sign * state::texture_tangent_v(tex_tangent_index)
        + n.z * state::normal());
    return gltf_normal_texture_lookup_value(valid: sample.valid, value: normal);
}

)mdl"
                                                         : "";
    const std::string module_source = fmt::format(
        "mdl 1.9;\nusing ::nvidia::sdk_examples::gltf_support import *;\n\n"
        "{}"
        "gltf_texture_lookup_value gltf_green_to_red(gltf_texture_lookup_value v) = "
        "gltf_texture_lookup_value(valid: v.valid, value: float4(v.value.y, v.value.y, v.value.y, v.value.w));\n\n"
        "gltf_texture_lookup_value falcor_gltf_anisotropy_texture(gltf_texture_lookup_value v, uniform float sign) = "
        "gltf_texture_lookup_value(valid: v.valid, value: float4(v.value.x, "
        "0.5 + sign * (v.value.y - 0.5), v.value.z, v.value.w));\n\n"
        "export material ImportedMaterial(\n{}\n) = "
        "{}(\n{}\n);\n",
        normal_helper,
        fmt::join(declarations, ",\n"),
        specgloss ? "gltf_material_khr_specular_glossiness" : "gltf_material",
        fmt::join(arguments, ",\n")
    );
    // Resource paths, source material names, factor values and UV indices never
    // enter the module identity. Class-compiled HLSL is shared later by MDLMaterial.
    const std::string module = "gltf_" + sgl::SHA1(module_source).hex_digest();
    properties.set("mdl_source", module_source);
    properties.set("mdl_material_name", "::" + module + "::ImportedMaterial");
    properties.set("mdl_library_path", (platform::runtime_directory() / "mdl").string());
    properties.set("mdl_class_compilation", true);
    // The SDK glTF definition reads this name; bind it to the imported color
    // semantic without imposing glTF naming on unrelated MDL materials.
    properties.set("mdl_scene_data:COLOR_0", classify_geometry_attribute(MeshAttributeSemantic::color, 0).fixed_id);
    properties.set("_scene_material_type", "MDLMaterial");
    // Thin transmission retains authored sidedness. Only volume boundaries
    // require both faces independently of doubleSided.
    properties.set("enable_backface_culling", thin && !params.get("doubleSided", false));
    return properties;
}

} // namespace falcor
