// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_falcor_material.h"

#include "material_network.h"
#include "falcor2/core/error.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/material_conversions.h"
#include "falcor2/render/material_types.h"

#include <fmt/format.h>

#include <filesystem>
#include <string_view>

namespace falcor {
namespace {

using importer_material::Node;
using importer_material::NodeInput;
using importer_material::NodeNetwork;

constexpr std::string_view FALCOR_TERMINAL = "_terminal:falcor:surface";
constexpr std::string_view STANDARD_MATERIAL_ID = "FalcorStandardMaterial";
constexpr std::string_view UNLIT_MATERIAL_ID = "FalcorUnlitMaterial";
constexpr std::string_view OPENPBR_MATERIAL_ID = "FalcorOpenPBRMaterial";

AlphaMode parse_alpha_mode(const ImporterMaterial& material, std::string_view value)
{
    if (value == "opaque")
        return AlphaMode::opaque;
    if (value == "mask")
        return AlphaMode::mask;
    if (value == "blend")
        return AlphaMode::blend;
    FALCOR_THROW("Falcor USD material '{}': alpha_mode must be opaque, mask, or blend; got '{}'", material.name, value);
}

template<typename T>
void copy_optional_input(const Properties& inputs, Properties& result, std::string_view name)
{
    if (auto value = inputs.get_optional<T>(name, true))
        result.set(name, *value);
}

struct Terminal {
    const std::vector<Properties>* nodes;
    std::string_view shader_id;
};

void set_texture_path(
    Properties& result,
    const Node& terminal_node,
    std::string_view path_property_name,
    bool srgb = true
)
{
    const NodeInput input = terminal_node.get_input(path_property_name);
    if (!input)
        return;

    const std::filesystem::path path = input.property().get<std::filesystem::path>();
    if (path.empty())
        return;

    importer_material::set_texture_path(result, path_property_name, path, srgb);
}

void copy_optional_openpbr_float_input(
    const Properties& inputs,
    Properties& result,
    const Node& terminal_node,
    std::string_view name
)
{
    copy_optional_input<float>(inputs, result, name);
    copy_optional_input<uint>(inputs, result, fmt::format("{}_texture_channel", name));
    set_texture_path(result, terminal_node, fmt::format("{}_texture_path", name), false);
}

void copy_optional_openpbr_color_input(
    const Properties& inputs,
    Properties& result,
    const Node& terminal_node,
    std::string_view name
)
{
    copy_optional_input<float3>(inputs, result, name);
    set_texture_path(result, terminal_node, fmt::format("{}_texture_path", name));
}

std::optional<Terminal> get_terminal(const ImporterMaterial& material)
{
    auto network_it = material.output_to_material_network.find(FALCOR_TERMINAL);
    if (network_it == material.output_to_material_network.end())
        return {};

    const auto& nodes = network_it->second;
    if (nodes.size() != 1) {
        FALCOR_THROW(
            "Falcor USD material '{}': expected exactly one node in {}, got {}",
            material.name,
            FALCOR_TERMINAL,
            nodes.size()
        );
    }

    const std::string_view shader_id = nodes.back().get<std::string_view>("info:id");
    if (shader_id != STANDARD_MATERIAL_ID && shader_id != UNLIT_MATERIAL_ID && shader_id != OPENPBR_MATERIAL_ID) {
        FALCOR_THROW(
            "Falcor USD material '{}': unsupported shader id '{}' on {}",
            material.name,
            shader_id,
            FALCOR_TERMINAL
        );
    }
    return Terminal{&nodes, shader_id};
}

Properties convert_terminal(const ImporterMaterial& material, const Terminal& terminal)
{
    const bool is_standard = terminal.shader_id == STANDARD_MATERIAL_ID;
    const bool is_openpbr = terminal.shader_id == OPENPBR_MATERIAL_ID;
    NodeNetwork network(terminal.nodes);
    const Node terminal_node = network.terminal();

    for (const NodeInput& input : terminal_node) {
        if (input.connection()) {
            FALCOR_THROW(
                "Falcor USD material '{}': input '{}' must be a value, not a connection",
                material.name,
                input.name()
            );
        }
    }

    const Properties& inputs = terminal_node.props();
    Properties result;
    if (is_openpbr) {
        result.set("_scene_material_type", "OpenPBRMaterial");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "base_weight");
        copy_optional_openpbr_color_input(inputs, result, terminal_node, "base_color");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "base_diffuse_roughness");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "base_metalness");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "specular_weight");
        copy_optional_openpbr_color_input(inputs, result, terminal_node, "specular_color");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "specular_roughness");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "specular_roughness_anisotropy");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "specular_ior");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "coat_weight");
        copy_optional_openpbr_color_input(inputs, result, terminal_node, "coat_color");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "coat_roughness");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "coat_roughness_anisotropy");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "coat_ior");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "coat_darkening");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "fuzz_weight");
        copy_optional_openpbr_color_input(inputs, result, terminal_node, "fuzz_color");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "fuzz_roughness");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "transmission_weight");
        copy_optional_openpbr_color_input(inputs, result, terminal_node, "transmission_color");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "transmission_depth");
        copy_optional_openpbr_color_input(inputs, result, terminal_node, "transmission_scatter");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "transmission_scatter_anisotropy");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "thin_film_weight");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "thin_film_thickness");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "thin_film_ior");
        copy_optional_openpbr_float_input(inputs, result, terminal_node, "emission_luminance");
        copy_optional_openpbr_color_input(inputs, result, terminal_node, "emission_color");
        copy_optional_input<float>(inputs, result, "opacity_factor");
        copy_optional_input<float>(inputs, result, "opacity_threshold");
        copy_optional_input<uint>(inputs, result, "opacity_texture_channel");
        copy_optional_input<float>(inputs, result, "normal_texture_scale");
        copy_optional_input<bool>(inputs, result, "geometry_thin_walled");
        set_texture_path(result, terminal_node, "opacity_texture_path", false);
        set_texture_path(result, terminal_node, "normal_texture_path", false);
        return result;
    }

    result.set("_scene_material_type", is_standard ? "StandardMaterial" : "UnlitMaterial");
    copy_optional_input<float>(inputs, result, "alpha_cutoff");
    copy_optional_input<float>(inputs, result, "alpha_factor");
    if (auto value = inputs.get_optional<std::string_view>("alpha_mode", true))
        result.set("alpha_mode", parse_alpha_mode(material, *value));
    copy_optional_input<float3>(inputs, result, "base_color_factor");
    copy_optional_input<bool>(inputs, result, "double_sided");
    set_texture_path(result, terminal_node, "base_color_texture_path");

    if (is_standard) {
        copy_optional_input<float3>(inputs, result, "emissive_factor");
        copy_optional_input<float>(inputs, result, "ior");
        copy_optional_input<float>(inputs, result, "metallic_factor");
        copy_optional_input<uint>(inputs, result, "metallic_texture_channel");
        copy_optional_input<float>(inputs, result, "roughness_factor");
        copy_optional_input<uint>(inputs, result, "roughness_texture_channel");
        copy_optional_input<float>(inputs, result, "diffuse_transmission_factor");
        copy_optional_input<float>(inputs, result, "specular_transmission_factor");
        copy_optional_input<bool>(inputs, result, "thin_walled");
        copy_optional_input<float3>(inputs, result, "transmission_factor");
        set_texture_path(result, terminal_node, "emissive_texture_path");
        set_texture_path(result, terminal_node, "metallic_roughness_texture_path", false);
        set_texture_path(result, terminal_node, "normal_texture_path", false);
        set_texture_path(result, terminal_node, "transmission_texture_path");
    }

    return result;
}

std::optional<Properties> convert_material_type(const ImporterMaterial& material, std::string_view expected_shader_id)
{
    const auto terminal = get_terminal(material);
    if (!terminal || terminal->shader_id != expected_shader_id)
        return {};
    return convert_terminal(material, *terminal);
}

} // namespace

std::optional<Properties> falcor_to_standardmaterial(const ImporterMaterial& importer_material)
{
    return convert_material_type(importer_material, STANDARD_MATERIAL_ID);
}

std::optional<Properties> falcor_to_unlitmaterial(const ImporterMaterial& importer_material)
{
    return convert_material_type(importer_material, UNLIT_MATERIAL_ID);
}

std::optional<Properties> falcor_to_openpbrmaterial(const ImporterMaterial& importer_material)
{
    return convert_material_type(importer_material, OPENPBR_MATERIAL_ID);
}

} // namespace falcor
