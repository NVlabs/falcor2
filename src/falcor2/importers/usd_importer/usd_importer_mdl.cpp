// SPDX-License-Identifier: Apache-2.0

#include "material_network.h"
#include "usd_importer_mdl.h"
#include "usd_importer_macros.h"

#include "falcor2/importers/importer_types.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace falcor {

namespace {
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
        mdl_library_path = source_asset->root_path().string();
        auto relative_path = source_asset->relative_path();
        mdl_material_name = (relative_path.parent_path() / relative_path.stem()).string();
    } else {
        // If it wasn't resolved in Usd, we just say it must be relative.
        mdl_library_path = "./";
        mdl_material_name = source_asset->stem().string();
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
    }

    result.set("mdl_class_compilation", true);
    result.set("_scene_material_type", "MDLMaterial");

    return result;
}

} // namespace falcor
