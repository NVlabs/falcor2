// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "codegen_wrapper_1_39.h"

#include "mx139/codegen.h"

#include "falcor2/core/enum.h"
#include "falcor2/core/error.h"
#include "falcor2/core/asset_resolver.h"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXGenShader/Util.h>

#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

namespace falcor {
namespace materialx {
namespace materialx_1_39 {

namespace mx = MaterialX;

namespace {

// clang-format off
const std::unordered_map<std::string, std::vector<char>> k_channels_character_vector = {
    {"float", {'x', 'x', 'x', 'x'}},
    {"color3", {'r', 'g', 'b', 'b'}},
    {"color4", {'r', 'g', 'b', 'a'}},
    {"vector2", {'x', 'y', '0', '0'}},
    {"vector3", {'x', 'y', 'z', 'z'}},
    {"vector4", {'x', 'y', 'z', 'w'}},
};

const std::unordered_map<std::string, size_t> k_channels_pattern_length = {
    {"float", 1},
    {"color3", 3},
    {"color4", 4},
    {"vector2", 2},
    {"vector3", 3},
    {"vector4", 4},
};
// clang-format on

bool connect_to_original_source(const mx::InputPtr& input, const mx::InputPtr& source_input)
{
    if (source_input->hasNodeName()) {
        input->setNodeName(source_input->getNodeName());
        if (source_input->hasOutputString())
            input->setOutputString(source_input->getOutputString());
        return true;
    }

    if (source_input->hasNodeGraphString()) {
        input->setNodeGraphString(source_input->getNodeGraphString());
        if (source_input->hasOutputString())
            input->setOutputString(source_input->getOutputString());
        return true;
    }

    if (source_input->hasOutputString()) {
        input->setOutputString(source_input->getOutputString());
        return true;
    }

    return false;
}

bool make_compatible(const mx::PortElementPtr& dst, const std::string& src_type)
{
    if (dst->getType() == src_type)
        return false;

    auto input = std::dynamic_pointer_cast<mx::Input>(dst);
    if (!input)
        return false;

    if (!k_channels_pattern_length.contains(dst->getType()))
        return false;

    if (!k_channels_character_vector.contains(src_type))
        return false;

    auto node = input->getParent()->asA<mx::Node>();
    if (!node)
        return false;
    auto graph = node->getParent()->asA<mx::GraphElement>();
    if (!graph)
        return false;

    const bool extract_scalar = dst->getType() == "float" && src_type != "float";
    mx::NodePtr compatibility_node = graph->addNode(
        extract_scalar ? "extract" : "convert",
        graph->createValidChildName(
            node->getName() + "_" + input->getName() + (extract_scalar ? "_extract" : "_convert")
        ),
        dst->getType()
    );
    mx::InputPtr compatibility_input = compatibility_node->addInput("in", src_type);
    if (!connect_to_original_source(compatibility_input, input))
        return false;
    if (extract_scalar)
        compatibility_node->addInput("index", "integer")->setValueString("0");

    input->removeAttribute(mx::PortElement::NODE_GRAPH_ATTRIBUTE);
    input->removeAttribute(mx::PortElement::OUTPUT_ATTRIBUTE);
    input->setNodeName(compatibility_node->getName());
    return true;
}

std::string get_connected_source_type(const mx::PortElementPtr& port)
{
    mx::NodePtr connected_node = port->getConnectedNode();
    if (!connected_node)
        return {};

    const std::string& output_string = port->getOutputString();
    if (output_string.empty())
        return connected_node->getType();

    mx::OutputPtr output;
    if (port->hasNodeName()) {
        if (const auto& node_def = connected_node->getNodeDef())
            output = node_def->getOutput(output_string);
    } else if (port->hasNodeGraphString()) {
        const auto& node_graph_name = port->getNodeGraphString();
        auto scope = port->getRoot();
        auto node_graph = scope->getChildOfType<mx::NodeGraph>(port->getQualifiedName(node_graph_name));
        if (!node_graph)
            node_graph = scope->getChildOfType<mx::NodeGraph>(node_graph_name);
        if (node_graph)
            output = node_graph->getOutput(output_string);
    } else {
        output = port->getDocument()->getOutput(output_string);
    }

    return output ? output->getType() : std::string();
}

bool adjust_channels(const mx::PortElementPtr& port)
{
    const std::string src_type = get_connected_source_type(port);
    if (src_type.empty())
        return false;

    return make_compatible(port, src_type);
}

void collect_ports(const mx::ElementPtr& element, std::vector<mx::PortElementPtr>& ports)
{
    for (auto& child : element->getChildren()) {
        if (auto port = std::dynamic_pointer_cast<mx::PortElement>(child))
            ports.push_back(port);
        else
            collect_ports(child, ports);
    }
}

bool make_ports_compatible(const mx::ElementPtr& element)
{
    std::vector<mx::PortElementPtr> ports;
    collect_ports(element, ports);

    bool adjusted = false;
    for (const auto& port : ports)
        adjusted |= adjust_channels(port);
    return adjusted;
}

bool load_libraries_if_present(
    mx::DocumentPtr doc,
    const std::filesystem::path& search_root,
    const mx::FilePathVec& folders
)
{
    if (!std::filesystem::exists(search_root))
        return false;

    mx::StringSet loaded = mx::loadLibraries(folders, mx::FileSearchPath(search_root.string()), doc);
    return !loaded.empty();
}

void import_standard_libraries(mx::DocumentPtr doc)
{
    bool loaded = false;

#if defined(MATERIALX_1_39_STD_LIBRARY_PATH)
    loaded |= load_libraries_if_present(doc, MATERIALX_1_39_STD_LIBRARY_PATH, {"libraries", "."});
#endif

#if defined(FALCOR_PROJECT_DIR)
    if (!loaded) {
        const std::filesystem::path materialx_root
            = std::filesystem::path(FALCOR_PROJECT_DIR) / "external" / "MaterialX";
        loaded |= load_libraries_if_present(doc, materialx_root, {"libraries"});
    }
#endif
}

void import_mx139_libraries(mx::DocumentPtr doc)
{
#if defined(MX139_MTLX_LIBRARY_PATH)
    load_libraries_if_present(doc, MX139_MTLX_LIBRARY_PATH, {"."});
#endif
}

void validate_optimize_graph_flags(const CodeGenDesc& desc)
{
    const uint32_t raw_flags = static_cast<uint32_t>(desc.optimize_graph_flags);
    const uint32_t known_flags = static_cast<uint32_t>(OptimizeGraphFlags::all);
    const uint32_t unknown_flags = raw_flags & ~known_flags;
    FALCOR_CHECK(
        unknown_flags == 0,
        "Unsupported Mx139 optimize graph flag bits 0x{:x}. Expected flags: closure_pruning, "
        "static_scatter_mode, fresnel_policy.",
        unknown_flags
    );
}

void validate_layering_mode(const CodeGenDesc& desc)
{
    switch (desc.layering_mode) {
    case LayeringMode::closure_tree:
    case LayeringMode::bsdf_mix:
        return;
    }
    FALCOR_THROW(
        "Unsupported Mx139 layering mode '{}'. Expected one of: closure_tree, bsdf_mix.",
        sgl::enum_to_string(desc.layering_mode)
    );
}

void validate_compensation_mode(const CodeGenDesc& desc)
{
    switch (desc.compensation_mode) {
    case CompensationMode::turquin_lut:
    case CompensationMode::turquin_analytic:
    case CompensationMode::no_compensation:
        return;
    }
    FALCOR_THROW(
        "Unsupported Mx139 compensation mode '{}'. Expected one of: turquin_lut, "
        "turquin_analytic, no_compensation.",
        sgl::enum_to_string(desc.compensation_mode)
    );
}

void append_asset_resolver_search_paths(mx::FileSearchPath& search_path, const CodeGenDesc& desc)
{
    if (!desc.asset_resolver)
        return;

    for (const std::filesystem::path& root : desc.asset_resolver->get_search_paths())
        search_path.append(mx::FilePath(root.generic_string()));
}

std::filesystem::path resolve_document_path(const CodeGenDesc& desc, const std::filesystem::path& material_path)
{
    if (desc.asset_resolver) {
        std::filesystem::path resolved_path = desc.asset_resolver->resolve_path(material_path);
        FALCOR_CHECK(!resolved_path.empty(), "MaterialX file `{}` does not exist.", material_path.string());
        return resolved_path;
    }

    std::filesystem::path resolved_path = std::filesystem::absolute(material_path).lexically_normal();
    FALCOR_CHECK(std::filesystem::exists(resolved_path), "MaterialX file `{}` does not exist.", material_path.string());
    return resolved_path;
}

mx::FileSearchPath make_document_search_path(const CodeGenDesc& desc, const std::filesystem::path& material_path)
{
    mx::FileSearchPath search_path;
    search_path.append(mx::FilePath(material_path.parent_path().generic_string()));
    append_asset_resolver_search_paths(search_path, desc);
    return search_path;
}

mx::FileSearchPath make_string_search_path(const CodeGenDesc& desc)
{
    mx::FileSearchPath search_path;
    append_asset_resolver_search_paths(search_path, desc);
    return search_path;
}

std::vector<RenderableElement> discover_renderable_elements_impl(mx::DocumentPtr doc)
{
    std::vector<RenderableElement> elements;
    for (const mx::TypedElementPtr& element : mx::findRenderableElements(doc))
        elements.push_back(RenderableElement{element->getNamePath(), element->getType()});

    return elements;
}

} // namespace

mx::DocumentPtr CodeGen_1_39::load_document(const CodeGenDesc& desc)
{
    validate_optimize_graph_flags(desc);
    validate_layering_mode(desc);
    validate_compensation_mode(desc);

    mx::DocumentPtr doc = mx::createDocument();
    if (auto str_ptr = std::get_if<std::string>(&desc.document)) {
        mx::FileSearchPath search_path = make_string_search_path(desc);
        mx::readFromXmlString(doc, *str_ptr, search_path);
    } else {
        std::filesystem::path material_path
            = resolve_document_path(desc, std::get<std::filesystem::path>(desc.document));
        mx::FileSearchPath search_path = make_document_search_path(desc, material_path);
        mx::readFromXmlFile(doc, mx::FilePath(material_path.generic_string()), search_path);
    }

    import_standard_libraries(doc);
    import_mx139_libraries(doc);
    make_ports_compatible(doc);

    std::string validation_message;
    if (!doc->validate(&validation_message))
        FALCOR_THROW("MaterialX 1.39 validation failed with message:\n{}", validation_message);

    return doc;
}

std::unique_ptr<CodeGenResult> CodeGen_1_39::generate(const CodeGenDesc& desc)
{
    mx::DocumentPtr doc = load_document(desc);
    return ::falcor::materialx::mx139::generate_code(doc, desc);
}

std::vector<RenderableElement> CodeGen_1_39::discover_renderable_elements(const CodeGenDesc& desc)
{
    return discover_renderable_elements_impl(load_document(desc));
}

} // namespace materialx_1_39
} // namespace materialx
} // namespace falcor
