// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_network.h"
#include "usd_importer_macros.h"
#include "usd_importer_materialx.h"
#include "falcor2/importers/importer_types.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/tf/stringUtils.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/reference.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>
END_DISABLE_USD_WARNINGS

#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/XmlIo.h>

namespace falcor {
namespace usd_importer {

namespace {

using namespace pxr;
namespace mx = MaterialX;

std::string sanitize_material_name(std::string name)
{
    const size_t colon_count = std::count(name.begin(), name.end(), ':');
    if (colon_count == 0)
        return name;

    std::string sanitized;
    sanitized.reserve(name.size() + colon_count);
    for (char ch : name) {
        (ch == ':') ? (void)sanitized.append("__") : (void)sanitized.push_back(ch);
    }

    return sanitized;
}

std::filesystem::path resolve_path(const UsdPrim& prim, const std::string& asset_path)
{
    for (const auto& prim_spec : prim.GetPrimStack()) {
        if (!prim_spec)
            continue;
        const SdfLayerHandle& layer = prim_spec->GetLayer();
        if (!layer)
            continue;

        const std::string absolute_path = layer->ComputeAbsolutePath(asset_path);
        if (!absolute_path.empty())
            return absolute_path;
    }

    return asset_path;
}

std::vector<std::string> get_materialx_material_names(const std::filesystem::path& file_path)
{
    std::vector<std::string> material_names;
    if (file_path.empty())
        return material_names;

    mx::FileSearchPath search_path;
    if (!file_path.parent_path().empty())
        search_path.append(mx::FilePath(file_path.parent_path().string()));

    mx::DocumentPtr document = mx::createDocument();
    mx::readFromXmlFile(document, mx::FilePath(file_path.string()), search_path);

    for (const mx::NodePtr& material_node : document->getMaterialNodes()) {
        if (!material_node)
            continue;
        if (material_node->getName().empty())
            continue;
        material_names.push_back(material_node->getName());
    }

    return material_names;
}

} // namespace


size_t process_mtlx_references(const pxr::UsdStageRefPtr& stage)
{
    using namespace pxr;
    static const TfToken MTLX_SOURCE_TYPE("mtlx");
    static const TfToken SHADER_OUTPUT_NAME("out");

    if (!stage)
        return 0;

    size_t generated_materials = 0;

    for (const UsdPrim& prim : UsdPrimRange::Stage(stage)) {
        if (!prim.IsA<UsdGeomScope>())
            continue;

        SdfReferenceVector authored_references;
        for (const SdfPrimSpecHandle& prim_spec : prim.GetPrimStack()) {
            if (!prim_spec || !prim_spec->HasReferences())
                continue;

            const SdfReferenceVector references = prim_spec->GetReferenceList().GetAppliedItems();
            authored_references.insert(authored_references.end(), references.begin(), references.end());
        }
        if (authored_references.empty())
            continue;

        const SdfPath scope_path = prim.GetPath();
        const SdfPath materials_scope_path = scope_path.AppendChild(TfToken("Materials"));
        for (const SdfReference& reference : authored_references) {
            const std::string& asset_path = reference.GetAssetPath();
            if (asset_path.empty())
                continue;
            if (!TfStringEndsWith(asset_path, ".mtlx"))
                continue;

            std::vector<std::string> material_names;
            const std::filesystem::path resolved_asset_path = resolve_path(prim, asset_path);
            try {
                material_names = get_materialx_material_names(resolved_asset_path);
            } catch (const mx::Exception& e) {
                sgl::log_warn(
                    "Failed to parse MaterialX reference '{}' (resolved '{}'): {}",
                    asset_path,
                    resolved_asset_path.string(),
                    e.what()
                );
                continue;
            }
            if (material_names.empty())
                continue;

            UsdGeomScope::Define(stage, materials_scope_path);
            for (const std::string& material_node_name : material_names) {
                const std::string material_name = sanitize_material_name(material_node_name);
                const SdfPath material_path = materials_scope_path.AppendChild(TfToken(material_name));

                UsdShadeMaterial material = UsdShadeMaterial::Define(stage, material_path);
                UsdShadeShader shader = UsdShadeShader::Define(stage, material_path.AppendChild(TfToken("Shader")));

                shader.SetSourceAsset(SdfAssetPath(asset_path), MTLX_SOURCE_TYPE);
                shader.SetSourceAssetSubIdentifier(TfToken(material_node_name), MTLX_SOURCE_TYPE);

                UsdShadeOutput shader_output = shader.CreateOutput(SHADER_OUTPUT_NAME, SdfValueTypeNames->Token);
                UsdShadeOutput material_surface = material.CreateSurfaceOutput(MTLX_SOURCE_TYPE);
                material_surface.ConnectToSource(
                    shader.ConnectableAPI(),
                    shader_output.GetBaseName(),
                    UsdShadeAttributeType::Output
                );

                ++generated_materials;
            }
        }
    }

    return generated_materials;
}

} // namespace usd_importer


std::optional<Properties> usd_to_mtlx(const ImporterMaterial& importer_material)
{
    auto nodes_it = importer_material.output_to_material_network.find("_terminal:mtlx:surface");
    if (nodes_it == importer_material.output_to_material_network.end())
        return {};

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
    const auto& source_asset = params.get_optional<std::filesystem::path>("info:mtlx:sourceAsset");
    const auto& source_asset_subidentifier = params.get_optional<std::string>("info:mtlx:sourceAsset:subIdentifier");
    if (!source_asset || !source_asset_subidentifier) {
        sgl::log_warn("Material {} is missing sourceAsset or sourceAsset:subIdentifier.", importer_material.name);
        return {};
    }

    // Split the path: basepath is the directory, mtlx_path is just the filename
    std::filesystem::path mtlx_basepath = source_asset->parent_path();
    std::filesystem::path mtlx_path = source_asset->filename();

    result.set("mtlx_basepath", mtlx_basepath);
    result.set("mtlx_path", mtlx_path);
    result.set("mtlx_node_name", source_asset_subidentifier.value());

    using namespace importer_material;
    NodeNetwork network(&nodes);
    const Node terminal = network.terminal();
    for (const auto& input : terminal) {
        result.copy_property(input.property());
    }

    result.set("_scene_material_type", "MaterialXMaterial");

    return result;
}

} // namespace falcor
