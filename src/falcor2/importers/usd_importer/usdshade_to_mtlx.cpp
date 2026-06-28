// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usdshade_to_mtlx.h"

#include "material_network.h"
#include "falcor2/importers/importer_types.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include <fmt/core.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <MaterialXFormat/Environ.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>

namespace falcor {

// NOTE: Parts of this conversion logic are adapted from OpenUSD's hdMtlx
// material network translation approach. Keep this attribution alongside
// SPDX metadata to make provenance and licensing intent explicit.

namespace {

namespace mx = MaterialX;
using namespace importer_material;

std::string get_material_name_from_path(const std::string& material_path)
{
    const size_t pos = material_path.find_last_of('/');
    if (pos == std::string::npos || pos + 1 >= material_path.size())
        return material_path;
    return material_path.substr(pos + 1);
}

std::string get_mx_node_string(const mx::NodeDefPtr& mx_node_def)
{
    return mx_node_def->hasNamespace() ? mx_node_def->getNamespace() + ":" + mx_node_def->getNodeString()
                                       : mx_node_def->getNodeString();
}

std::string convert_to_mtlx_type(std::string_view usd_type_name)
{
    static const std::unordered_map<std::string_view, std::string_view> TYPE_TABLE = {
        {"bool", "boolean"},
        {"int", "integer"},
        {"float", "float"},
        {"color3f", "color3"},
        {"color4f", "color4"},
        {"float2", "vector2"},
        {"float3", "vector3"},
        {"float4", "vector4"},
        {"matrix3d", "matrix33"},
        {"matrix4d", "matrix44"},
        {"asset", "filename"},
        {"string", "string"},
    };

    if (auto it = TYPE_TABLE.find(usd_type_name); it != TYPE_TABLE.end())
        return std::string(it->second);
    return {};
}

std::string value_to_mtlx_string(const Properties::iterator& it)
{
    switch (it.type()) {
    case PropertyType::bool_:
        return it.get<bool>() ? "true" : "false";
    case PropertyType::int_:
        return fmt::format("{}", it.get<int64_t>());
    case PropertyType::float_:
        return fmt::format("{}", it.get<float>());
    case PropertyType::float2: {
        const float2 v = it.get<float2>();
        return fmt::format("{}, {}", v.x, v.y);
    }
    case PropertyType::float3: {
        const float3 v = it.get<float3>();
        return fmt::format("{}, {}, {}", v.x, v.y, v.z);
    }
    case PropertyType::float4: {
        const float4 v = it.get<float4>();
        return fmt::format("{}, {}, {}, {}", v.x, v.y, v.z, v.w);
    }
    case PropertyType::float3x3: {
        const float3x3 m = it.get<float3x3>();
        std::string result;
        for (uint32_t r = 0; r < 3; ++r) {
            for (uint32_t c = 0; c < 3; ++c) {
                if (r != 0 || c != 0)
                    result += ", ";
                result += fmt::format("{}", m[r][c]);
            }
        }
        return result;
    }
    case PropertyType::float4x4: {
        const float4x4 m = it.get<float4x4>();
        std::string result;
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < 4; ++c) {
                if (r != 0 || c != 0)
                    result += ", ";
                result += fmt::format("{}", m[r][c]);
            }
        }
        return result;
    }
    case PropertyType::string:
        return it.get<std::string>();
    default:
        break;
    }
    return {};
}

mx::FileSearchPath compute_search_paths()
{
    mx::FileSearchPath search_paths;

#ifdef MATERIALX_1_39_STD_LIBRARY_PATH
    search_paths.append(mx::FilePath(MATERIALX_1_39_STD_LIBRARY_PATH));
#endif

    auto get_env_var = [](const char* env_name) -> std::string
    {
#ifdef _WIN32
        char* buffer = nullptr;
        size_t len = 0;
        if (_dupenv_s(&buffer, &len, env_name) != 0 || !buffer)
            return {};
        std::string value(buffer);
        free(buffer);
        return value;
#else
        const char* value = std::getenv(env_name);
        return value ? std::string(value) : std::string();
#endif
    };

    auto append_env_paths = [&search_paths, &get_env_var](const char* env_name)
    {
        const std::string env_value = get_env_var(env_name);
        if (env_value.empty())
            return;

#ifdef _WIN32
        constexpr char SEP = ';';
#else
        constexpr char SEP = ':';
#endif

        std::string value(env_value);
        size_t start = 0;
        while (start < value.size()) {
            const size_t end = value.find(SEP, start);
            const std::string part = end == std::string::npos ? value.substr(start) : value.substr(start, end - start);
            if (!part.empty())
                search_paths.append(mx::FilePath(part));
            if (end == std::string::npos)
                break;
            start = end + 1;
        }
    };

    append_env_paths("PXR_MTLX_PLUGIN_SEARCH_PATHS");
    append_env_paths("PXR_MTLX_STDLIB_SEARCH_PATHS");
    return search_paths;
}

bool load_libraries_from_search_path(
    const mx::DocumentPtr& libs,
    const mx::FileSearchPath& search_paths,
    const mx::FilePathVec& library_folders
)
{
    if (search_paths.asString().empty())
        return false;

    mx::loadLibraries(library_folders, search_paths, libs);
    return !libs->getNodeDefs().empty();
}

const mx::DocumentPtr& get_std_libraries()
{
    static const mx::DocumentPtr std_libs = []()
    {
        mx::DocumentPtr libs = mx::createDocument();
        const mx::FileSearchPath search_paths = compute_search_paths();
        load_libraries_from_search_path(libs, search_paths, {"libraries", "."});

#if defined(FALCOR_PROJECT_DIR)
        if (libs->getNodeDefs().empty()) {
            const std::filesystem::path materialx_root
                = std::filesystem::path(FALCOR_PROJECT_DIR) / "external" / "MaterialX";
            if (std::filesystem::exists(materialx_root))
                load_libraries_from_search_path(libs, mx::FileSearchPath(materialx_root.string()), {"libraries"});
        }
#endif

        if (libs->getNodeDefs().empty()) {
            throw std::runtime_error("Failed to load MaterialX 1.39 standard libraries for USD MaterialX conversion.");
        }
        return libs;
    }();
    return std_libs;
}

std::string get_input_type(const mx::NodeDefPtr& mx_node_def, const NodeInput& input)
{
    if (!input.type().empty()) {
        const std::string mapped = convert_to_mtlx_type(input.type());
        if (!mapped.empty())
            return mapped;
    }

    if (mx_node_def) {
        if (mx::InputPtr mx_input = mx_node_def->getActiveInput(std::string(input.name())))
            return mx_input->getType();
    }

    return {};
}

class MtlxBuilder {
public:
    MtlxBuilder(const ImporterMaterial& material)
        : m_material(material)
    {
        m_doc = mx::createDocument();
        m_doc->importLibrary(get_std_libraries());
    }

    std::optional<Properties> build()
    {
        auto nodes_it = m_material.output_to_material_network.find("_terminal:mtlx:surface");
        if (nodes_it == m_material.output_to_material_network.end() || nodes_it->second.empty())
            return {};

        NodeNetwork network(&nodes_it->second);
        const Node terminal_node = network.terminal();
        if (!terminal_node)
            return {};

        if (terminal_node.props().get<std::string_view>("info:implementationSource", "") != "id")
            return {};

        const std::string_view terminal_id = terminal_node.props().get<std::string_view>("info:id", "");
        if (terminal_id.empty())
            return {};

        mx::NodeDefPtr terminal_def = m_doc->getNodeDef(std::string(terminal_id));
        if (!terminal_def)
            return {};

        m_node_graph = m_doc->addNodeGraph(m_doc->createValidChildName("NG"));

        const std::string material_name = get_material_name_from_path(m_material.name);
        const std::string shader_category = get_mx_node_string(terminal_def);
        mx::NodePtr shader_node = m_doc->addNode(shader_category, "Surface", "surfaceshader");
        if (!shader_node)
            return {};

        mx::NodePtr material_node = m_doc->addMaterialNode(m_doc->createValidChildName(material_name), shader_node);
        if (!material_node)
            return {};

        fill_node_parameters(terminal_node, terminal_def, shader_node);
        connect_node_inputs(terminal_node, shader_node);

        std::string validation_message;
        if (!m_doc->validate(&validation_message)) {
            sgl::log_warn("Generated MaterialX validation warnings for {}: {}", m_material.name, validation_message);
        }

        // Serialize the MaterialX document to a string
        std::string mtlx_xml = mx::writeToXmlString(m_doc);

        Properties result;
        result.set("mtlx_buffer", mtlx_xml);
        result.set("mtlx_node_name", material_node->getName());
        result.set("_scene_material_type", "MaterialXMaterial");
        return result;
    }

private:
    void fill_node_parameters(const Node& node, const mx::NodeDefPtr& node_def, const mx::NodePtr& mx_node)
    {
        for (const NodeInput& input : node) {
            if (input.connection())
                continue;

            const std::string value_string = value_to_mtlx_string(input.property());
            const std::string input_type = get_input_type(node_def, input);
            mx::InputPtr mx_input = mx_node->setInputValue(std::string(input.name()), value_string, input_type);
            if (mx_input && input.colorspace().has_value())
                mx_input->setColorSpace(std::string(input.colorspace().value()));
        }
    }

    mx::NodePtr add_or_get_node(const Node& node)
    {
        const std::string path = node.props().get<std::string>("_path", "");
        if (path.empty())
            return nullptr;

        if (auto it = m_added_nodes.find(path); it != m_added_nodes.end())
            return it->second;

        const std::string_view node_id = node.props().get<std::string_view>("info:id", "");
        mx::NodeDefPtr node_def = m_doc->getNodeDef(std::string(node_id));
        if (!node_def)
            node_def = m_doc->getNodeDef("ND_surface");
        if (!node_def)
            return nullptr;

        // Create a node name from the path; let MaterialX make it valid
        std::string node_name = path;
        const size_t last_slash = node_name.find_last_of('/');
        if (last_slash != std::string::npos && last_slash + 1 < node_name.size())
            node_name = node_name.substr(last_slash + 1);
        if (node_name.empty())
            node_name = "node";

        mx::NodePtr mx_node = m_node_graph->addNode(get_mx_node_string(node_def), node_name, node_def->getType());
        if (!mx_node)
            return nullptr;

        if (mx_node->getNodeDefString().empty() && !node_id.empty())
            mx_node->setNodeDefString(std::string(node_id));

        fill_node_parameters(node, node_def, mx_node);

        m_added_nodes.emplace(path, mx_node);

        connect_node_inputs(node, mx_node);
        return mx_node;
    }

    void connect_node_inputs(const Node& node, const mx::NodePtr& mx_node)
    {
        for (const NodeInput& input : node) {
            if (!input.connection())
                continue;

            const Node& source = input.connection()->source();
            mx::NodePtr upstream = add_or_get_node(source);
            if (!upstream)
                continue;

            mx::InputPtr mx_input = mx_node->getInput(std::string(input.name()));
            if (!mx_input)
                mx_input = mx_node->addInput(std::string(input.name()), upstream->getType());

            if (mx_input)
                mx_input->setConnectedNode(upstream);
        }
    }

private:
    const ImporterMaterial& m_material;
    mx::DocumentPtr m_doc;
    mx::NodeGraphPtr m_node_graph;
    std::unordered_map<std::string, mx::NodePtr> m_added_nodes;
};

} // namespace

std::optional<Properties> usdshade_to_mtlx(const ImporterMaterial& importer_material)
{
    return MtlxBuilder(importer_material).build();
}

} // namespace falcor
