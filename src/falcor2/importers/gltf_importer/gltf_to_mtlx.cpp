// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gltf_conversions.h"
#include "gltf_conversion_utils.h"
#include "falcor2/render/scene_types.h"
#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>
#include <cmath>

namespace falcor {
namespace {
namespace mx = MaterialX;

void value(const mx::NodePtr& node, std::string_view name, std::string_view type, const std::string& text)
{
    node->addInput(std::string(name), std::string(type))->setValueString(text);
}
void connect(const mx::NodePtr& node, std::string_view name, const mx::NodePtr& input)
{
    node->addInput(std::string(name), input->getType())->setConnectedNode(input);
}
std::string numbers(float2 v)
{
    return fmt::format("{}, {}", v.x, v.y);
}
std::string numbers(float3 v)
{
    return fmt::format("{}, {}, {}", v.x, v.y, v.z);
}
} // namespace

std::optional<Properties> gltf_to_mtlx(
    const ImporterMaterial& source,
    std::span<const ImporterTexture> textures,
    bool use_vertex_color,
    UVOrigin uv_origin,
    const std::function<std::filesystem::path(const ImporterTexture&)>& resolve_texture
)
{
    const auto& p = source.params;
    const auto type = p.get<std::string>("_type", "");
    if (type != "gltf_pbrMetallicRoughness" && type != "gltf_pbrSpecularGlossiness")
        return {};
    const bool sg = type == "gltf_pbrSpecularGlossiness";
    Properties r;
    r.set("_scene_material_type", "MaterialXMaterial");
    r.set("enable_backface_culling", !p.get("doubleSided", false) && p.get("thicknessFactor", 0.f) == 0.f);
    auto doc = mx::createDocument();
    doc->setColorSpace("lin_rec709");
    auto surface = doc->addNode("gltf_pbr", "gltf_surface", "surfaceshader");
    size_t serial = 0;
    auto node = [&](std::string_view category, std::string_view type)
    {
        return doc->addNode(std::string(category), fmt::format("n{}", serial++), std::string(type));
    };
    auto constant = [&](std::string_view type, const std::string& text)
    {
        auto n = node("constant", type);
        value(n, "value", type, text);
        return n;
    };
    auto binary = [&](std::string_view op, const mx::NodePtr& a, const mx::NodePtr& b)
    {
        auto n = node(op, a->getType());
        connect(n, "in1", a);
        connect(n, "in2", b);
        return n;
    };
    auto component = [&](const mx::NodePtr& a, int index)
    {
        auto n = node("extract", "float");
        connect(n, "in", a);
        value(n, "index", "integer", std::to_string(index));
        return n;
    };
    auto rgb = [&](const mx::NodePtr& a)
    {
        auto n = node("convert", "color3");
        connect(n, "in", a);
        return n;
    };
    auto image = [&](std::string_view input, bool color) -> mx::NodePtr
    {
        auto index = p.get_optional<int>(input);
        if (!index)
            return {};
        FALCOR_CHECK(*index >= 0 && size_t(*index) < textures.size(), "Invalid glTF texture index for '{}'", input);
        const auto& texture = textures[*index];
        const int uv_index = p.get(fmt::format("{}:texcoord", input), 0);
        FALCOR_CHECK(uv_index >= 0 && uv_index < 8, "Invalid glTF UV set {} for '{}'", uv_index, input);
        auto uv = node("texcoord", "vector2");
        value(uv, "index", "integer", std::to_string(uv_index));
        if (p.get(fmt::format("{}:has_transform", input), false)) {
            auto flip = [&](mx::NodePtr uv)
            {
                return binary("add", binary("multiply", uv, constant("vector2", "1, -1")), constant("vector2", "0, 1"));
            };
            if (uv_origin == UVOrigin::lower_left)
                uv = flip(uv);
            uv = binary(
                "multiply",
                uv,
                constant("vector2", numbers(p.get(fmt::format("{}:scale", input), float2(1.f))))
            );
            auto rotated = node("rotate2d", "vector2");
            connect(rotated, "in", uv);
            // MaterialX rotate2d and glTF both use clockwise rotation in Y-down UVs.
            value(
                rotated,
                "amount",
                "float",
                fmt::format("{}", p.get(fmt::format("{}:rotation", input), 0.f) * (180.f / 3.14159265358979323846f))
            );
            uv = binary(
                "add",
                rotated,
                constant("vector2", numbers(p.get(fmt::format("{}:offset", input), float2(0.f))))
            );
            if (uv_origin == UVOrigin::lower_left)
                uv = flip(uv);
        }
        auto n = node("image", color ? "color4" : "vector4");
        value(n, "file", "filename", resolve_texture(texture).generic_string());
        n->getInput("file")->setColorSpace(color ? "srgb_texture" : "lin_rec709");
        value(n, "default", n->getType(), "1, 1, 1, 1");
        connect(n, "texcoord", uv);
        const auto wrap = [](TextureWrapMode mode)
        {
            return mode == TextureWrapMode::clamp_to_edge ? "clamp"
                : mode == TextureWrapMode::mirror_repeat  ? "mirror"
                                                          : "periodic";
        };
        value(n, "uaddressmode", "string", wrap(texture.wrap_s));
        value(n, "vaddressmode", "string", wrap(texture.wrap_t));
        value(n, "filtertype", "string", texture.min_filter == TextureFilterMode::nearest ? "closest" : "linear");
        if (texture.min_filter != texture.mag_filter)
            gltf_conversion::warn(
                r,
                source,
                "MaterialX",
                fmt::format("{} uses different min/mag filters; graph uses the minification filter", input)
            );
        return n;
    };
    auto scalar
        = [&](std::string_view output, std::string_view factor, float fallback, std::string_view texture, int channel)
    {
        auto n = constant("float", fmt::format("{}", p.get(factor, fallback)));
        if (auto tex = image(texture, false))
            n = binary("multiply", n, component(tex, channel));
        connect(surface, output, n);
    };
    auto color = [&](std::string_view output, std::string_view factor, float3 fallback, std::string_view texture)
    {
        auto n = constant("color3", numbers(p.get(factor, fallback)));
        if (auto tex = image(texture, true))
            n = binary("multiply", n, rgb(tex));
        connect(surface, output, n);
    };

    const auto base = p.get(sg ? "diffuseFactor" : "baseColorFactor", float4(1.f));
    auto base_rgb = constant("color3", numbers(base.xyz()));
    auto alpha = constant("float", fmt::format("{}", base.w));
    if (auto tex = image(sg ? "diffuseTexture" : "baseColorTexture", true)) {
        base_rgb = binary("multiply", base_rgb, rgb(tex));
        alpha = binary("multiply", alpha, component(tex, 3));
    }
    if (use_vertex_color) {
        auto vertex = node("geomcolor", "color4");
        value(vertex, "index", "integer", "0");
        base_rgb = binary("multiply", base_rgb, rgb(vertex));
        alpha = binary("multiply", alpha, component(vertex, 3));
    }
    connect(surface, "base_color", base_rgb);
    connect(surface, "alpha", alpha);
    const auto alpha_mode = p.get("alphaMode", AlphaMode::opaque);
    value(
        surface,
        "alpha_mode",
        "integer",
        alpha_mode == AlphaMode::mask        ? "1"
            : alpha_mode == AlphaMode::blend ? "2"
                                             : "0"
    );
    value(surface, "alpha_cutoff", "float", fmt::format("{}", p.get("alphaCutoff", 0.5f)));

    if (sg) {
        value(surface, "metallic", "float", "0");
        auto gloss = constant("float", fmt::format("{}", p.get("glossinessFactor", 1.f)));
        auto specular = constant("color3", numbers(p.get("specularFactor", float3(1.f)) * 25.f));
        if (auto tex = image("specularGlossinessTexture", true)) {
            gloss = binary("multiply", gloss, component(tex, 3));
            specular = binary("multiply", specular, rgb(tex));
        }
        connect(surface, "roughness", binary("subtract", constant("float", "1"), gloss));
        connect(surface, "specular_color", specular);
        gltf_conversion::warn(
            r,
            source,
            "MaterialX",
            "SpecGloss approximated with the existing glTF MR definition and dielectric F0 tint"
        );
    } else {
        scalar("metallic", "metallicFactor", 1.f, "metallicRoughnessTexture", 2);
        scalar("roughness", "roughnessFactor", 1.f, "metallicRoughnessTexture", 1);
        scalar("specular", "specularFactor", 1.f, "specularTexture", 3);
        color("specular_color", "specularColorFactor", float3(1.f), "specularColorTexture");
    }
    value(surface, "ior", "float", fmt::format("{}", p.get("ior", 1.5f)));
    color("emissive", "emissiveFactor", float3(0.f), "emissiveTexture");
    scalar("transmission", "transmissionFactor", 0.f, "transmissionTexture", 0);
    scalar("clearcoat", "clearcoatFactor", 0.f, "clearcoatTexture", 0);
    scalar("clearcoat_roughness", "clearcoatRoughnessFactor", 0.f, "clearcoatRoughnessTexture", 1);
    color("sheen_color", "sheenColorFactor", float3(0.f), "sheenColorTexture");
    scalar("sheen_roughness", "sheenRoughnessFactor", 0.f, "sheenRoughnessTexture", 3);
    scalar("iridescence", "iridescenceFactor", 0.f, "iridescenceTexture", 0);
    value(surface, "iridescence_ior", "float", fmt::format("{}", p.get("iridescenceIor", 1.3f)));
    const float film_min = p.get("iridescenceThicknessMinimum", 100.f);
    const float film_max = p.get("iridescenceThicknessMaximum", 400.f);
    auto film = constant("float", fmt::format("{}", film_max));
    if (auto tex = image("iridescenceThicknessTexture", false))
        film = binary(
            "add",
            constant("float", fmt::format("{}", film_min)),
            binary("multiply", constant("float", fmt::format("{}", film_max - film_min)), component(tex, 1))
        );
    connect(surface, "iridescence_thickness", film);
    auto anisotropy = constant("float", fmt::format("{}", p.get("anisotropyStrength", 0.f)));
    auto anisotropy_rotation = constant("float", fmt::format("{}", p.get("anisotropyRotation", 0.f)));
    if (auto tex = image("anisotropyTexture", false)) {
        anisotropy = binary("multiply", anisotropy, component(tex, 2));
        auto direction = node("atan2", "float");
        for (auto [name, channel] : {std::pair{"inx", 0}, std::pair{"iny", 1}})
            connect(
                direction,
                name,
                binary(
                    "subtract",
                    binary("multiply", component(tex, channel), constant("float", "2")),
                    constant("float", "1")
                )
            );
        anisotropy_rotation = binary("add", anisotropy_rotation, direction);
    }
    connect(surface, "anisotropy_strength", anisotropy);
    connect(surface, "anisotropy_rotation", anisotropy_rotation);
    for (auto [input, output] :
         {std::pair{"normalTexture", "normal"}, std::pair{"clearcoatNormalTexture", "clearcoat_normal"}}) {
        if (auto tex = image(input, false)) {
            auto channels = node("convert", "vector3");
            connect(channels, "in", tex);
            auto normal = node("normalmap", "vector3");
            connect(normal, "in", channels);
            value(normal, "scale", "float", fmt::format("{}", p.get(fmt::format("{}:strength", input), 1.f)));
            connect(surface, output, normal);
        }
    }
    scalar("thickness", "thicknessFactor", 0.f, "thicknessTexture", 1);
    value(surface, "attenuation_color", "color3", numbers(p.get("attenuationColor", float3(1.f))));
    if (auto distance = p.get_optional<float>("attenuationDistance"); distance && std::isfinite(*distance))
        value(surface, "attenuation_distance", "float", fmt::format("{}", *distance));
    gltf_conversion::unsupported(source, r, "MaterialX", {"hasDiffuseTransmission"});
    if (p.get("hasTransmission", false) || p.get("hasVolume", false))
        gltf_conversion::warn(
            r,
            source,
            "MaterialX",
            "Transmission/volume use existing MaterialX transport; thin-boundary and medium semantics need "
            "qualification"
        );
    auto material = doc->addNode("surfacematerial", "gltf_material", "material");
    connect(material, "surfaceshader", surface);
    r.set("mtlx_buffer", mx::writeToXmlString(doc));
    r.set("mtlx_node_name", "gltf_material");
    return r;
}

} // namespace falcor
