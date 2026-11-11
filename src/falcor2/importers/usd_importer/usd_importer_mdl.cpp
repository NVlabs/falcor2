// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_network.h"
#include "usd_importer_mdl.h"
#include "usd_importer_macros.h"

#include "falcor2/importers/importer_types.h"

#include <sgl/core/bitmap.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace falcor {

namespace {

std::optional<bool> detect_texture_srgb(const std::filesystem::path& path)
{
    if (path.empty() || !std::filesystem::is_regular_file(path))
        return {};
    // Match Kit's observed automatic policy using the decoded image's original component type.
    // Documented metadata policy and bit-depth/channel fallback:
    // https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials_color_space.html
    const sgl::Bitmap bitmap(path);
    return bitmap.srgb_gamma() && bitmap.component_type() == sgl::Bitmap::ComponentType::uint8
        && (bitmap.channel_count() == 3 || bitmap.channel_count() == 4);
}

std::pair<std::optional<std::filesystem::path>, std::optional<std::string>>
get_mdl_source_asset(const Properties& params)
{
    // obtain the path
    auto source_asset = params.get_optional<std::filesystem::path>("info:mdl:sourceAsset");
    auto source_asset_subidentifier = params.get_optional<std::string>("info:mdl:sourceAsset:subIdentifier");

    // Try to get the path from deprecated names
    if (!source_asset && !source_asset_subidentifier) {
        source_asset = params.get_optional<std::filesystem::path>("info:sourceAsset");
        source_asset_subidentifier = params.get_optional<std::string>("name");
    }

    return {source_asset, source_asset_subidentifier};
}
} // namespace

std::optional<Properties> usd_to_mdl(const ImporterMaterial& importer_material)
{
    auto nodes_it = importer_material.output_to_material_network.find("_terminal:mdl:surface");
    if (nodes_it == importer_material.output_to_material_network.end()) {
        // Legacy files can have MDL on the surface output as well. Check if it is so.
        // If not, we fail here silently.
        nodes_it = importer_material.output_to_material_network.find("_terminal:surface");
        if (nodes_it == importer_material.output_to_material_network.end())
            return {};
        const std::vector<Properties>& nodes = nodes_it->second;
        if (nodes.empty())
            return {};
        const Properties& params = nodes.back();
        // obtain the path
        auto [source_asset, source_asset_subidentifier] = get_mdl_source_asset(params);
        if (!source_asset || source_asset->extension() != ".mdl")
            return {};
    }

    const std::vector<Properties>& nodes = nodes_it->second;
    if (nodes.size() != 1) {
        sgl::log_warn(
            "Material {} has unsupported network size of {}. Only supporting 1 node networks for now.",
            importer_material.name,
            nodes.size()
        );
        return {};
    }

    const Properties& params = nodes[0];

    Properties result;
    // obtain the path
    auto [source_asset, source_asset_subidentifier] = get_mdl_source_asset(params);
    if (!source_asset || !source_asset_subidentifier) {
        sgl::log_warn("Material {} is missing sourceAsset or sourceAsset:subIdentifier.", importer_material.name);
        return {};
    }

    std::string mdl_library_path;
    std::string mdl_material_name;
    if (params.get<bool>("info:mdl:sourceAsset:is_resolved")) {
        mdl_library_path = source_asset->root_path().string() + ";" + source_asset->parent_path().string();
        auto relative_path = source_asset->relative_path();
        mdl_material_name = (relative_path.parent_path() / relative_path.stem()).string();
    } else {
        // Bare library references retain their package. Explicit filesystem references must resolve in USD.
        const std::string authored = source_asset->generic_string();
        if (source_asset->has_root_path() || authored.starts_with("./") || authored.starts_with("../")
            || authored.find(':') != std::string::npos || authored.starts_with('\\')) {
            sgl::log_warn("Material {} has unresolved MDL source asset '{}'.", importer_material.name, authored);
            return {};
        }
        mdl_material_name = (source_asset->parent_path() / source_asset->stem()).string();
    }
    // Convert path separators to ::.
    {
        std::string converted;
        converted.reserve(mdl_material_name.size() + 2);
        for (char c : mdl_material_name) {
            if (c == '/' || c == '\\')
                converted += "::";
            else
                converted += c;
        }
        mdl_material_name = std::move(converted);
    }
    // Prepend :: if not already there.
    if (!mdl_material_name.starts_with("::"))
        mdl_material_name = "::" + mdl_material_name;
    mdl_material_name = mdl_material_name + "::" + source_asset_subidentifier.value();

    result.set("mdl_library_path", mdl_library_path);
    result.set("mdl_material_name", mdl_material_name);

    using namespace importer_material;
    NodeNetwork network(&nodes);
    const Node terminal = network.terminal();
    for (const auto& input : terminal) {
        result.copy_property(input.property());
        // MDL path properties use a separate sRGB flag; preserve the USD asset's authored transfer function.
        if (input.type() == "asset" && input.colorspace()) {
            if (*input.colorspace() == "sRGB")
                result.set(fmt::format("{}:srgb", input.name()), true);
            else if (*input.colorspace() == "raw")
                result.set(fmt::format("{}:srgb", input.name()), false);
            else if (*input.colorspace() == "auto") {
                const auto path = input.property().get<std::filesystem::path>();
                if (const auto srgb = detect_texture_srgb(path))
                    result.set(fmt::format("{}:srgb", input.name()), *srgb);
            }
        }
    }

    result.set("mdl_class_compilation", true);
    result.set("_scene_material_type", "MDLMaterial");

    return result;
}

} // namespace falcor
