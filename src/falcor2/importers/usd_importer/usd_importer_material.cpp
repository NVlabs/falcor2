// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_macros.h"
#include "usd_importer_material.h"
#include "material_network.h"
#include "usd_importer_utils.h"
#include "falcor2/importers/importer_types.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/tf/pathUtils.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdr/declare.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/usd/sdr/shaderProperty.h>
#include <pxr/usd/usdShade/udimUtils.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/nodeDefAPI.h>
#include <pxr/usd/usdLux/lightFilter.h>
#include <pxr/usd/usdLux/lightAPI.h>
END_DISABLE_USD_WARNINGS

#include <array>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <vector>

namespace falcor {

namespace usd_importer {


namespace {

using namespace pxr;

// NOTE: The build_material/walk_graph material network extraction logic below
// is adapted in part from OpenUSD's UsdImaging material graph code in
// pxr/usdImaging/usdImaging/materialParamUtils.cpp

// We need to find the first layer that changes the value
// of the parameter so that we anchor relative paths to that.
static SdfLayerHandle find_layer_handle(const UsdAttribute& attr, const UsdTimeCode& time)
{
    for (const auto& spec : attr.GetPropertyStack(time)) {
        if (spec->HasDefaultValue() || spec->GetLayer()->GetNumTimeSamplesForPath(spec->GetPath()) > 0) {
            return spec->GetLayer();
        }
    }
    return TfNullPtr;
}

// Resolve symlinks for string path.
// Resolving symlinks can reduce the number of unique textures added into the
// texture registry since it may use the asset path as hash.
static bool resolve_symlinks(const std::string& srcPath, std::string* outPath)
{
    std::string error;
    *outPath = TfRealPath(srcPath, false, &error);

    if (outPath->empty() || !error.empty()) {
        return false;
    }

    return true;
}

// Resolve symlinks for asset path.
// Resolving symlinks can reduce the number of unique textures added into the
// texture registry since it may use the asset path as hash.
static SdfAssetPath resolve_asset_symlinks(const SdfAssetPath& assetPath)
{
    std::string p = assetPath.GetResolvedPath();
    if (p.empty()) {
        p = assetPath.GetAssetPath();
    }

    if (resolve_symlinks(p, &p)) {
        return SdfAssetPath(assetPath.GetAssetPath(), p);
    } else {
        return assetPath;
    }
}

static SdfAssetPath
resolve_asset_attribute(const SdfAssetPath& assetPath, const UsdAttribute& attr, const UsdTimeCode& time)
{
    TRACE_FUNCTION();

    // Not a UDIM, resolve symlinks and exit.
    if (!UsdShadeUdimUtils::IsUdimIdentifier(assetPath.GetAssetPath())) {
        return resolve_asset_symlinks(assetPath);
    }

    const std::string resolvedPath
        = UsdShadeUdimUtils::ResolveUdimPath(assetPath.GetAssetPath(), find_layer_handle(attr, time));
    // If the path doesn't resolve, return the input path
    if (resolvedPath.empty()) {
        return assetPath;
    }
    return SdfAssetPath(assetPath.GetAssetPath(), resolvedPath);
}

pxr::VtValue
resolve_material_param_value(const pxr::UsdAttribute& attr, const UsdTimeCode& time = UsdTimeCode::EarliestTime())
{
    using namespace pxr;

    VtValue value;
    if (!attr.Get(&value, time)) {
        return value;
    }

    if (!value.IsHolding<SdfAssetPath>()) {
        return value;
    }

    return VtValue(resolve_asset_attribute(value.UncheckedGet<SdfAssetPath>(), attr, time));
}

static TfToken get_primvar_name_attribute_value(
    const SdrShaderNodeConstPtr& sdrNode,
    const Properties& params,
    const TfToken& propName
)
{
    VtValue vtName;

    // If the name of the primvar was authored in parameters list.
    // The authored value is the strongest opinion

    const auto& paramIt = params.find(propName.GetText());
    if (paramIt != params.end()) {
        vtName = TfToken(paramIt->get<std::string>());
    }

    // If we didn't find an authored value consult Sdr for the default value.

    if (vtName.IsEmpty() && sdrNode) {
        if (SdrShaderPropertyConstPtr sdrPrimvarInput = sdrNode->GetShaderInput(propName)) {
            vtName = sdrPrimvarInput->GetDefaultValue();
        }
    }

    if (vtName.IsHolding<TfToken>()) {
        return vtName.UncheckedGet<TfToken>();
    } else if (vtName.IsHolding<std::string>()) {
        return TfToken(vtName.UncheckedGet<std::string>());
    }

    return TfToken();
}

static TfToken get_node_id(
    const UsdShadeConnectableAPI& shadeNode,
    const TfTokenVector& shader_source_types,
    const TfTokenVector& render_contexts
)
{
    UsdShadeNodeDefAPI nodeDef(shadeNode.GetPrim());
    if (nodeDef) {
        // Extract the identifier of the node.
        // GetShaderNodeForSourceType will try to find/create an Sdr node for
        // all three info cases: info:id, info:sourceAsset and info:sourceCode.
        TfToken id;
        if (!nodeDef.GetShaderId(&id)) {
            for (const auto& sourceType : shader_source_types) {
                if (SdrShaderNodeConstPtr sdrNode = nodeDef.GetShaderNodeForSourceType(sourceType)) {
                    return sdrNode->GetIdentifier();
                }
            }
        }
        return id;
    }

    // If the node is a light filter that doesn't have a NodeDefAPI, then we
    // try to get the light shader ID from the light filter for the given
    // render contexts.
    UsdLuxLightFilter lightFilter(shadeNode);
    if (lightFilter) {
        TfToken id = lightFilter.GetShaderId(render_contexts);
        if (!id.IsEmpty()) {
            return id;
        }
    } else {
        // Otherwise, if the node is a light that doesn't have a NodeDefAPI,
        // then we try to get the light shader ID from the light for the given
        // render contexts.
        UsdLuxLightAPI light(shadeNode);
        if (light) {
            TfToken id = light.GetShaderId(render_contexts);
            if (!id.IsEmpty()) {
                return id;
            }
        }
    }

    // Otherwise for connectable nodes that don't implement NodeDefAPI and we
    // fail to get a light shader ID for, the type name of the prim is used as
    // the node's identifier. This will currently always be the case for prims
    // like light filters.
    return shadeNode.GetPrim().GetTypeName();
}


using PathSet = std::unordered_set<pxr::SdfPath, pxr::SdfPath::Hash>;

void walk_graph(
    ImporterMaterial* material,
    std::vector<Properties>* nodes,
    PathSet* visited_nodes,
    const pxr::UsdShadeConnectableAPI& shade_node,
    const pxr::TfTokenVector& shader_source_types,
    const pxr::TfTokenVector& render_contexts
)
{
    using namespace pxr;

    Properties params;
    params.set("_path", shade_node.GetPath().GetText());
    params.set("_type", shade_node.GetPrim().GetTypeName().GetText());
    if (!TF_VERIFY(shade_node.GetPath() != SdfPath::EmptyPath())) {
        return;
    }

    // Already processed the node.
    if (!visited_nodes->insert(shade_node.GetPath()).second) {
        return;
    }

    const std::vector<UsdShadeInput>& shade_node_inputs = shade_node.GetInputs();
    for (auto& input : shade_node_inputs) {
        std::string input_name = input.GetBaseName().GetString();

        const UsdShadeAttributeVector& attrs = input.GetValueProducingAttributes(false);

        for (const UsdAttribute& attr : attrs) {
            auto attr_type = UsdShadeUtils::GetType(attr.GetName());

            if (attr_type == UsdShadeAttributeType::Output) {
                walk_graph(
                    material,
                    nodes,
                    visited_nodes,
                    UsdShadeConnectableAPI(attr.GetPrim()),
                    shader_source_types,
                    render_contexts
                );

                /// Add relationship
                params.set(
                    fmt::format("{}:connect", input_name),
                    attr.GetPrim().GetPath().GetString() + "." + UsdShadeOutput(attr).GetBaseName().GetText()
                );
            } else if (attr_type == UsdShadeAttributeType::Input) {

                const VtValue value = resolve_material_param_value(attr);
                if (!value.IsEmpty()) {
                    set_from_value(params, input_name, value);
                }
            }
            if (attr.HasColorSpace()) {
                set_from_value(params, fmt::format("{}:colorspace", input_name), VtValue(attr.GetColorSpace()));
            }
            params.set(fmt::format("{}:typename", input_name), attr.GetTypeName().GetAsToken().GetText());
        }
    }

    UsdShadeNodeDefAPI node_def(shade_node.GetPrim());
    if (node_def) {
        {
            TfToken id;
            if (node_def.GetShaderId(&id)) {
                params.set("info:id", id.GetText());
            }
        }

        params.set("info:implementationSource", node_def.GetImplementationSource().GetText());

        for (auto& source_type_str : node_def.GetSourceTypes()) {
            const TfToken source_type = TfToken(source_type_str);
            SdfAssetPath path;
            if (node_def.GetSourceAsset(&path, source_type)) {
                path = resolve_asset_symlinks(path);
                if (path.GetResolvedPath().empty()) {
                    params.set(fmt::format("info:{}:sourceAsset", source_type_str), path.GetAssetPath());
                    params.set(fmt::format("info:{}:sourceAsset:is_resolved", source_type_str), false);
                } else {
                    params.set(fmt::format("info:{}:sourceAsset", source_type_str), path.GetResolvedPath());
                    params.set(fmt::format("info:{}:sourceAsset:is_resolved", source_type_str), true);
                }
            }

            TfToken sub_identifier;
            if (node_def.GetSourceAssetSubIdentifier(&sub_identifier, source_type)) {
                params.set(fmt::format("info:{}:sourceAsset:subIdentifier", source_type_str), sub_identifier.GetText());
            }

            std::string source_code;
            if (node_def.GetSourceCode(&source_code, source_type)) {
                params.set(fmt::format("info:{}:sourceCode", source_type_str), source_code);
            }
        }
    }

    nodes->push_back(std::move(params));
}

} // namespace

void build_material(
    ImporterMaterial& material,
    const pxr::UsdPrim& usd_terminal,
    const std::string_view& terminal_identifier,
    const pxr::TfTokenVector& shader_source_types,
    const pxr::TfTokenVector& render_contexts
)
{
    using namespace pxr;

    PathSet visited_nodes;
    std::vector<Properties> nodes;

    walk_graph(
        &material,
        &nodes,
        &visited_nodes,
        pxr::UsdShadeConnectableAPI(usd_terminal),
        shader_source_types,
        render_contexts
    );

    if (!TF_VERIFY(!nodes.empty()))
        return;

    material.output_to_material_network[std::string(terminal_identifier)] = std::move(nodes);
}

} // namespace usd_importer

namespace {
using namespace importer_material;

bool convert_texture_coords(const Node& usduvtexture_node, float4x4& st_transform)
{
    NodeInput st_input = usduvtexture_node.get_input("st");
    if (!st_input || !st_input.connection())
        return false;

    const Node source_node = st_input.connection()->source();
    std::string_view id = source_node.props().get<std::string_view>("info:id", "");
    if (id != "UsdTransform2d")
        return false;

    float4x4 S = float4x4::identity();
    float4x4 R = float4x4::identity();
    float4x4 T = float4x4::identity();

    if (auto param = source_node.props().get_optional<float2>("scale")) {
        S = sgl::math::matrix_from_scaling(float3(*param, 1.f));
    }

    if (auto param = source_node.props().get_optional<float>("rotation")) {
        R = sgl::math::matrix_from_rotation_z(sgl::math::radians(*param));
    }

    if (auto param = source_node.props().get_optional<float2>("translation")) {
        T = sgl::math::matrix_from_translation(float3(*param, 0.f));
    }

    st_transform = mul(mul(T, R), S);
    return true;
}


std::optional<ImporterTexture> to_texture(const NodeInput& input)
{
    if (!input.connection())
        return {};

    const Node source_node = input.connection()->source();
    auto id = source_node.props().get<std::string_view>("info:id", "");
    if (id != "UsdUVTexture")
        return {};

    ImporterTexture result;
    convert_texture_coords(source_node, result.texture_transform);

    result.min_filter = TextureFilterMode::linear;
    result.mag_filter = TextureFilterMode::linear;
    result.wrap_s = TextureWrapMode::repeat;
    result.wrap_t = TextureWrapMode::repeat;

    NodeInput file_input = source_node.get_input("file");
    if (!file_input || file_input.connection())
        return {};

    result.texture_path = file_input.property().get<std::filesystem::path>();

    if (input.colorspace().value_or("") == "sRGB")
        result.is_srgb = true;

    if (file_input.colorspace().value_or("") == "sRGB")
        result.is_srgb = true;

    result.source_name = input.connection()->output();

    return result;
}

} // namespace

std::optional<Properties> usdpreviewsurface_to_flattened(
    const ImporterMaterial& importer_material,
    std::function<int(const ImporterTexture&)> add_texture_to_scene
)
{
    auto nodes_it = importer_material.output_to_material_network.find("_terminal:surface");
    if (nodes_it == importer_material.output_to_material_network.end())
        return {};

    NodeNetwork network = NodeNetwork(&nodes_it->second);
    const Node terminal = network.terminal();
    if (!terminal || terminal.props().get<std::string_view>("info:id", "") != "UsdPreviewSurface")
        return {};

    Properties result;
    result.set("_type", "usd_UsdPreviewSurface");
    // set defaults
    result.set("diffuseColor", float3(0.18));
    result.set("emissiveColor", float3(0));
    result.set("useSpecularWorkflow", int(0));
    result.set("metallic", 0.f);
    result.set("roughness", 0.5f);
    result.set("clearcoat", 0.f);
    result.set("clearcoatRoughness", 0.01f);
    result.set("opacity", 1.f);
    result.set("opacityMode", "transparent"); // alternative: presence
    result.set("opacityThreshold", 0.f);
    result.set("ior", 1.5f);
    result.set("normal", float3(0, 0, 1));
    result.set("displacement", 0.f);
    result.set("occlusion", 1.f);

    for (const NodeInput& input : terminal) {

        if (!input.connection()) {
            result.copy_property(input.property());
            continue;
        }

        if (auto texture = to_texture(input)) {
            int texture_index = add_texture_to_scene(*texture);
            result.set(input.name(), texture_index);
        }
    }

    return result;
}


std::optional<Properties> usdpreviewsurface_to_standardmaterial(const ImporterMaterial& importer_material)
{
    auto channel_to_index = [&](const std::string_view& param_name, const std::string_view& channels)
    {
        if (channels.size() != 1) {
            sgl::log_warn(
                "Material {}: expected single channel for {}, got {}",
                importer_material.name,
                param_name,
                channels
            );
            return 0;
        }

        switch (channels[0]) {
        case 'r':
            return 0;
        case 'g':
            return 1;
        case 'b':
            return 2;
        case 'a':
            return 3;
        default:
            break;
        };

        sgl::log_warn(
            "Material {}: unexpected channel kind for {}, got {}, expected one of r,g,b,a",
            importer_material.name,
            param_name,
            channels
        );
        return 0;
    };

    std::map<ImporterTexture, int> texture_to_index;
    std::vector<const ImporterTexture*> textures;
    auto add_texture = [&](const ImporterTexture& texture) -> int
    {
        const int index = (int)textures.size();
        auto [it, inserted] = texture_to_index.try_emplace(texture, index);
        if (inserted)
            textures.push_back(&it->first);
        return it->second;
    };

    auto flattened = usdpreviewsurface_to_flattened(importer_material, add_texture);
    if (!flattened)
        return {};

    Properties properties;
    properties.set("_scene_material_type", "StandardMaterial");
    const Properties& params = *flattened;
    // Specular workflow not supported by StandardMaterial yet.
    if (params.get<int>("useSpecularWorkflow", 0) == 1)
        return {};

    if (auto it = params.find("diffuseColor"); it != params.end()) {
        switch (it.type()) {
        case PropertyType::int_:
            properties.set("base_color_texture_path", textures[it.get<int>()]->texture_path);
            break;
        case PropertyType::float4:
            properties.set("base_color_factor", it.get<float4>().xyz());
            break;
        case PropertyType::float3:
            properties.set("base_color_factor", it.get<float3>());
            break;
        case PropertyType::float_:
            properties.set("base_color_factor", float3(it.get<float>()));
            break;
        default:
            break;
        }
    }

    if (auto it = params.find("emissiveColor"); it != params.end()) {
        switch (it.type()) {
        case PropertyType::int_:
            properties.set("emissive_texture_path", textures[it.get<int>()]->texture_path);
            break;
        case PropertyType::float4:
            properties.set("emissive_factor", it.get<float4>().xyz());
            break;
        case PropertyType::float3:
            properties.set("emissive_factor", it.get<float3>());
            break;
        case PropertyType::float_:
            properties.set("emissive_factor", float3(it.get<float>()));
            break;
        default:
            break;
        }
    }

    auto metallic_it = params.find("metallic");
    auto roughness_it = params.find("roughness");

    if (metallic_it != params.end() && roughness_it != params.end()) {
        if (metallic_it.type() == PropertyType::int_ && roughness_it.type() == PropertyType::int_) {
            const ImporterTexture& metallic_texture = *textures[metallic_it.get<int>()];
            const ImporterTexture& roughness_texture = *textures[roughness_it.get<int>()];
            if (metallic_texture.texture_path == roughness_texture.texture_path) {
                properties.set("metallic_roughness_texture_path", metallic_texture.texture_path);
                properties.set("metallic_texture_channel", channel_to_index("metallic", metallic_texture.source_name));
                properties.set(
                    "roughness_texture_channel",
                    channel_to_index("roughness", roughness_texture.source_name)
                );
            } else {
                sgl::log_warn(
                    "Material {}: metallic ({}) and roughness ({}) come from different textures, currently not "
                    "supported by "
                    "StandardMaterial.",
                    importer_material.name,
                    textures[metallic_it.get<int>()]->texture_path,
                    textures[roughness_it.get<int>()]->texture_path
                );
            }
        }
    }

    if (metallic_it != params.end() && metallic_it.type() == PropertyType::float_) {
        properties.set("metallic_factor", metallic_it.get<float>());
    }

    if (roughness_it != params.end() && roughness_it.type() == PropertyType::float_) {
        properties.set("roughness_factor", roughness_it.get<float>());
    }

    if (auto it = params.find("normal"); it != params.end() && it.type() == PropertyType::int_) {
        properties.set("normal_texture_path", textures[it.get<int>()]->texture_path);
    }

    return properties;
}
std::optional<Properties> usdpreviewsurface_to_mtlx(const ImporterMaterial& importer_material)
{
    FALCOR_UNUSED(importer_material);
    return {};
}

} // namespace falcor
