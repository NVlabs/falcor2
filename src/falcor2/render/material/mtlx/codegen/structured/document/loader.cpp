// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "loader.h"

#include "../pipeline/request_validation.h"
#include "passes/default_geomprops.h"

#include "falcor2/core/asset_resolver.h"
#include "falcor2/core/error.h"
#include "falcor2/core/platform.h"
#include "falcor2/render/material_aov_registry.h"

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Node.h>
#include <MaterialXCore/Types.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenShader/Util.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

namespace {

// MaterialX XML permits connecting compatible vector/color widths. MaterialX
// generators normally insert these conversions before ShaderGraph creation;
// keep that document mutation in the structured document-loading boundary.
const std::unordered_map<std::string, std::vector<char>> CHANNEL_VECTORS = {
    {"float", {'x', 'x', 'x', 'x'}},
    {"color3", {'r', 'g', 'b', 'b'}},
    {"color4", {'r', 'g', 'b', 'a'}},
    {"vector2", {'x', 'y', '0', '0'}},
    {"vector3", {'x', 'y', 'z', 'z'}},
    {"vector4", {'x', 'y', 'z', 'w'}},
};

const std::unordered_map<std::string, size_t> CHANNEL_COUNTS = {
    {"float", 1},
    {"color3", 3},
    {"color4", 4},
    {"vector2", 2},
    {"vector3", 3},
    {"vector4", 4},
};

// Channel tables used when the requested root is a value output rather than a
// material. They mirror MaterialX's raster preview conventions so the
// compatibility wrapper preserves the legacy visible preview behavior.
const std::unordered_map<std::string, std::vector<char>> PREVIEW_CHANNEL_VECTORS = {
    {"boolean", {'x', 'x', 'x', 'x'}},
    {"integer", {'x', 'x', 'x', 'x'}},
    {"float", {'x', 'x', 'x', 'x'}},
    {"color3", {'r', 'g', 'b', 'b'}},
    {"color4", {'r', 'g', 'b', 'a'}},
    {"vector2", {'x', 'y', '0', '0'}},
    {"vector3", {'x', 'y', 'z', 'z'}},
    {"vector4", {'x', 'y', 'z', 'w'}},
};

const std::unordered_map<std::string, size_t> PREVIEW_CHANNEL_COUNTS = {
    {"boolean", 1},
    {"integer", 1},
    {"float", 1},
    {"color3", 3},
    {"color4", 4},
    {"vector2", 2},
    {"vector3", 3},
    {"vector4", 4},
};

const std::unordered_set<std::string> BLACK_PREVIEW_TYPES = {
    "boolean",
    "matrix33",
    "matrix44",
};

bool connect_to_original_source(const mx::InputPtr& input, const mx::InputPtr& source)
{
    if (source->hasNodeName()) {
        input->setNodeName(source->getNodeName());
        if (source->hasOutputString())
            input->setOutputString(source->getOutputString());
        return true;
    }
    if (source->hasNodeGraphString()) {
        input->setNodeGraphString(source->getNodeGraphString());
        if (source->hasOutputString())
            input->setOutputString(source->getOutputString());
        return true;
    }
    if (source->hasOutputString()) {
        input->setOutputString(source->getOutputString());
        return true;
    }
    return false;
}

std::string connected_source_type(const mx::PortElementPtr& port)
{
    mx::NodePtr node = port->getConnectedNode();
    if (!node)
        return {};
    if (!port->hasOutputString())
        return node->getType();

    mx::OutputPtr output;
    if (port->hasNodeName()) {
        if (mx::NodeDefPtr node_def = node->getNodeDef())
            output = node_def->getOutput(port->getOutputString());
    } else if (port->hasNodeGraphString()) {
        auto scope = port->getRoot();
        auto graph = scope->getChildOfType<mx::NodeGraph>(port->getQualifiedName(port->getNodeGraphString()));
        if (!graph)
            graph = scope->getChildOfType<mx::NodeGraph>(port->getNodeGraphString());
        if (graph)
            output = graph->getOutput(port->getOutputString());
    } else {
        output = port->getDocument()->getOutput(port->getOutputString());
    }
    return output ? output->getType() : std::string();
}

bool make_compatible(const mx::PortElementPtr& destination, const std::string& source_type)
{
    if (destination->getType() == source_type)
        return false;
    mx::InputPtr input = destination->asA<mx::Input>();
    if (!input || !CHANNEL_COUNTS.contains(destination->getType()) || !CHANNEL_VECTORS.contains(source_type))
        return false;
    mx::NodePtr node = input->getParent()->asA<mx::Node>();
    mx::GraphElementPtr graph = node && node->getParent() ? node->getParent()->asA<mx::GraphElement>() : nullptr;
    if (!graph)
        return false;

    const bool extract_scalar = destination->getType() == "float" && source_type != "float";
    mx::NodePtr conversion = graph->addNode(
        extract_scalar ? "extract" : "convert",
        graph->createValidChildName(
            node->getName() + "_" + input->getName() + (extract_scalar ? "_extract" : "_convert")
        ),
        destination->getType()
    );
    mx::InputPtr conversion_input = conversion->addInput("in", source_type);
    if (!connect_to_original_source(conversion_input, input))
        return false;
    if (extract_scalar)
        conversion->addInput("index", "integer")->setValueString("0");
    input->removeAttribute(mx::PortElement::NODE_GRAPH_ATTRIBUTE);
    input->removeAttribute(mx::PortElement::OUTPUT_ATTRIBUTE);
    input->setNodeName(conversion->getName());
    return true;
}

void collect_ports(const mx::ElementPtr& element, std::vector<mx::PortElementPtr>& ports)
{
    for (const mx::ElementPtr& child : element->getChildren()) {
        if (mx::PortElementPtr port = child->asA<mx::PortElement>())
            ports.push_back(port);
        else
            collect_ports(child, ports);
    }
}

// MaterialX documents may connect compatible scalar, vector, and color widths,
// while ShaderGraph construction expects the connected types to match. Insert
// explicit extract/convert nodes into the owning graph and reconnect each input
// through them. Snapshot the original ports first: this operation mutates the
// document tree, and newly inserted conversion ports must not be visited again.
void make_ports_compatible(const mx::DocumentPtr& document)
{
    std::vector<mx::PortElementPtr> ports;
    collect_ports(document, ports);
    for (const mx::PortElementPtr& port : ports) {
        const std::string source_type = connected_source_type(port);
        if (!source_type.empty())
            make_compatible(port, source_type);
    }
}

bool can_preview_output_as_diffuse(const std::string& type)
{
    return (PREVIEW_CHANNEL_VECTORS.contains(type) && PREVIEW_CHANNEL_COUNTS.contains(type))
        || BLACK_PREVIEW_TYPES.contains(type);
}

bool is_scalar_preview_type(const std::string& type)
{
    return type == "boolean" || type == "integer" || type == "float";
}

void connect_to_output(const mx::InputPtr& input, const mx::OutputPtr& output)
{
    if (auto node_graph = output->getParent()->asA<mx::NodeGraph>()) {
        if (mx::NodeDefPtr node_def = node_graph->getNodeDef()) {
            // A nodegraph with a nodedef is an implementation graph. Materialize
            // an instance so defaults and implementation selection are resolved
            // exactly as they are for an ordinary MaterialX node use.
            mx::DocumentPtr document = output->getDocument();
            mx::NodePtr instance = document->addNodeInstance(
                node_def,
                document->createValidChildName("mx_preview_" + output->getName() + "_instance")
            );
            input->setNodeName(instance->getName());
            if (node_graph->getOutputCount() > 1)
                input->setOutputString(output->getName());
            return;
        }
    }

    input->setConnectedOutput(output);
}

void connect_to_port_source(const mx::InputPtr& input, const mx::PortElementPtr& source)
{
    if (auto output = source->asA<mx::Output>()) {
        connect_to_output(input, output);
        return;
    }
    if (source->hasNodeName()) {
        input->setNodeName(source->getNodeName());
        if (source->hasOutputString())
            input->setOutputString(source->getOutputString());
        return;
    }
    if (source->hasNodeGraphString()) {
        input->setNodeGraphString(source->getNodeGraphString());
        if (source->hasOutputString())
            input->setOutputString(source->getOutputString());
    }
}

std::string diffuse_preview_channels(const std::string& source_type, const std::string& output_channels)
{
    std::string channels;
    channels.reserve(3);
    if (output_channels.empty()) {
        auto source = PREVIEW_CHANNEL_VECTORS.find(source_type);
        if (source == PREVIEW_CHANNEL_VECTORS.end())
            return {};
        for (size_t index = 0; index < 3; ++index)
            channels.push_back(source->second[index]);
        return channels;
    }

    for (size_t index = 0; index < 3; ++index)
        channels.push_back(output_channels[std::min(index, output_channels.size() - 1)]);
    return channels;
}

std::string separate_node_category(const std::string& source_type)
{
    auto count = PREVIEW_CHANNEL_COUNTS.find(source_type);
    FALCOR_CHECK(count != PREVIEW_CHANNEL_COUNTS.end(), "Cannot separate unsupported MaterialX type `{}`", source_type);
    return "separate" + std::to_string(count->second);
}

std::string separate_node_def(const std::string& source_type)
{
    return "ND_" + separate_node_category(source_type) + "_" + source_type;
}

std::string separated_output_name(const std::string& source_type, char channel)
{
    static const std::unordered_map<char, std::string> COLOR_OUTPUTS = {
        {'r', "outr"},
        {'g', "outg"},
        {'b', "outb"},
        {'a', "outa"},
    };
    static const std::unordered_map<char, std::string> VECTOR_OUTPUTS = {
        {'x', "outx"},
        {'y', "outy"},
        {'z', "outz"},
        {'w', "outw"},
    };

    const auto& outputs = source_type.starts_with("color") ? COLOR_OUTPUTS : VECTOR_OUTPUTS;
    auto output = outputs.find(channel);
    FALCOR_CHECK(
        output != outputs.end(),
        "MaterialX preview channel `{}` is not valid for source type `{}`",
        channel,
        source_type
    );
    return output->second;
}

mx::NodePtr
convert_output_to_color3(const mx::DocumentPtr& document, const std::string& prefix, const mx::OutputPtr& output)
{
    mx::NodePtr convert
        = document->addNode("convert", document->createValidChildName(prefix + "_convert_to_color3"), "color3");
    connect_to_output(convert->addInput("in", output->getType()), output);
    return convert;
}

mx::NodePtr compose_channel_preview_to_color3(
    const mx::DocumentPtr& document,
    const std::string& prefix,
    const mx::OutputPtr& output
)
{
    const std::string source_type = connected_source_type(output);
    FALCOR_CHECK(!source_type.empty(), "Cannot determine source type for MaterialX output `{}`", output->getNamePath());

    const std::string output_channels
        = output->hasAttribute("channels") ? output->getAttribute("channels") : std::string();
    const std::string channels = diffuse_preview_channels(source_type, output_channels);
    FALCOR_CHECK(
        !channels.empty(),
        "MaterialX output `{}` has no color3 preview channels from `{}`",
        output->getNamePath(),
        source_type
    );

    mx::NodePtr separate;
    if (!is_scalar_preview_type(source_type)) {
        separate = document->addNode(
            separate_node_category(source_type),
            document->createValidChildName(prefix + "_separate"),
            "multioutput"
        );
        separate->setNodeDefString(separate_node_def(source_type));
        connect_to_port_source(separate->addInput("in", source_type), output);
    }

    mx::NodePtr combine
        = document->addNode("combine3", document->createValidChildName(prefix + "_combine_color3"), "color3");
    for (size_t index = 0; index < 3; ++index) {
        mx::InputPtr input = combine->addInput("in" + std::to_string(index + 1), "float");
        const char channel = channels[index];
        if (channel == '0' || channel == '1') {
            input->setValueString(channel == '0' ? "0.0" : "1.0");
        } else if (is_scalar_preview_type(source_type)) {
            connect_to_port_source(input, output);
        } else {
            input->setNodeName(separate->getName());
            input->setOutputString(separated_output_name(source_type, channel));
        }
    }
    return combine;
}

void connect_diffuse_preview_color(
    const mx::InputPtr& color_input,
    const mx::DocumentPtr& document,
    const std::string& prefix,
    const mx::OutputPtr& output
)
{
    if (BLACK_PREVIEW_TYPES.contains(output->getType())) {
        // MaterialX's raster generators pack unsupported non-color preview
        // outputs to black. Preserve that visible behavior here.
        color_input->setValueString("0.0, 0.0, 0.0");
    } else if (output->hasAttribute("channels")) {
        color_input->setConnectedNode(compose_channel_preview_to_color3(document, prefix, output));
    } else if (output->getType() == "color3") {
        connect_to_output(color_input, output);
    } else {
        color_input->setConnectedNode(convert_output_to_color3(document, prefix, output));
    }
}

mx::NodePtr wrap_output_in_oren_nayar_preview(const mx::DocumentPtr& document, const mx::OutputPtr& output)
{
    FALCOR_CHECK(document, "Cannot wrap MaterialX output without a document");
    FALCOR_CHECK(output, "Cannot wrap null MaterialX value output");
    FALCOR_CHECK(
        can_preview_output_as_diffuse(output->getType()),
        "MaterialX output `{}` has type `{}` and cannot be previewed as an Oren-Nayar BSDF",
        output->getNamePath(),
        output->getType()
    );

    const std::string prefix = document->createValidChildName("mx_preview_" + output->getName());
    mx::NodePtr bsdf = document->addNode("oren_nayar_diffuse_bsdf", prefix + "_bsdf", "BSDF");
    connect_diffuse_preview_color(bsdf->addInput("color", "color3"), document, prefix, output);
    bsdf->addInput("roughness", "float")->setValueString("0.0");

    mx::NodePtr surface = document->addNode("surface", prefix + "_surface", mx::SURFACE_SHADER_TYPE_STRING);
    surface->setConnectedNode("bsdf", bsdf);

    mx::NodePtr material = document->addNode("surfacematerial", prefix + "_material", mx::MATERIAL_TYPE_STRING);
    material->setConnectedNode("surfaceshader", surface);
    return material;
}

mx::ElementPtr wrap_previewable_root_output(const mx::DocumentPtr& document, const mx::ElementPtr& root)
{
    mx::OutputPtr output = root ? root->asA<mx::Output>() : nullptr;
    return output && can_preview_output_as_diffuse(output->getType())
        ? wrap_output_in_oren_nayar_preview(document, output)
        : root;
}

std::vector<mx::TypedElementPtr> find_top_level_nodes_of_type(const mx::DocumentPtr& document, const std::string& type)
{
    std::vector<mx::TypedElementPtr> elements;
    for (const mx::ElementPtr& child : document->getChildren()) {
        if (!child->asA<mx::Node>())
            continue;
        mx::TypedElementPtr typed = child->asA<mx::TypedElement>();
        if (typed && typed->getType() == type)
            elements.push_back(typed);
    }
    return elements;
}

bool is_named_root_candidate(const mx::ElementPtr& element)
{
    mx::TypedElementPtr typed = element ? element->asA<mx::TypedElement>() : nullptr;
    if (!typed || (!element->asA<mx::Node>() && !element->asA<mx::Output>()))
        return false;
    const std::string type = typed->getType();
    return type == mx::MATERIAL_TYPE_STRING || type == mx::SURFACE_SHADER_TYPE_STRING;
}

mx::ElementPtr find_unique_root_element_by_local_name(const mx::DocumentPtr& document, const std::string& name)
{
    std::vector<mx::ElementPtr> matches;
    for (const mx::ElementPtr& element : document->traverseTree()) {
        if (element->getName() == name && is_named_root_candidate(element))
            matches.push_back(element);
    }

    if (matches.size() == 1)
        return matches.front();
    if (matches.empty())
        FALCOR_THROW("MTLX could not find MaterialX element '{}'.", name);

    std::string names = matches.front()->getNamePath();
    for (size_t index = 1; index < matches.size(); ++index)
        names += ", " + matches[index]->getNamePath();
    FALCOR_THROW("MTLX found multiple MaterialX elements named '{}' ({}); provide a qualified name path.", name, names);
}

std::vector<mx::TypedElementPtr> find_unconsumed_top_level_bsdfs(const mx::DocumentPtr& document)
{
    std::unordered_set<std::string> consumed_node_names;
    for (const mx::ElementPtr& child : document->getChildren()) {
        mx::NodePtr node = child->asA<mx::Node>();
        if (!node)
            continue;
        for (const mx::InputPtr& input : node->getInputs()) {
            if (input->hasNodeName())
                consumed_node_names.insert(input->getNodeName());
        }
    }

    std::vector<mx::TypedElementPtr> elements;
    for (const mx::TypedElementPtr& typed : find_top_level_nodes_of_type(document, mx::BSDF_TYPE_STRING)) {
        if (!consumed_node_names.contains(typed->getName()))
            elements.push_back(typed);
    }
    return elements;
}

// MaterialX names have no '*' characters. Match literal names and star patterns
// without introducing regex syntax or compiling expressions for exact lookups.
bool matches_output_pattern(std::string_view name, std::string_view pattern)
{
    size_t n = 0, p = 0, star = std::string_view::npos, retry = 0;
    while (n < name.size()) {
        if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            retry = n;
        } else if (p < pattern.size() && pattern[p] == name[n]) {
            ++p;
            ++n;
        } else if (star != std::string_view::npos) {
            p = star + 1;
            n = ++retry;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*')
        ++p;
    return p == pattern.size();
}

mx::ElementPtr find_root_element(const mx::DocumentPtr& document, const CodeGenDesc& desc)
{
    if (!desc.node_name.empty()) {
        if (mx::ElementPtr element = document->getDescendant(desc.node_name))
            return element;
        return find_unique_root_element_by_local_name(document, desc.node_name);
    }

    std::vector<mx::TypedElementPtr> elements = mx::findRenderableElements(document);
    if (elements.size() == 1)
        return elements.front();
    if (elements.empty()) {
        elements = find_top_level_nodes_of_type(document, mx::SURFACE_SHADER_TYPE_STRING);
        if (elements.size() == 1)
            return elements.front();
    }
    if (elements.empty()) {
        elements = find_unconsumed_top_level_bsdfs(document);
        if (elements.size() == 1)
            return elements.front();
    }
    if (elements.empty())
        FALCOR_THROW("MTLX could not automatically find a renderable MaterialX element.");

    std::string names = elements.front()->getNamePath();
    for (size_t index = 1; index < elements.size(); ++index)
        names += ", " + elements[index]->getNamePath();
    FALCOR_THROW("MTLX found multiple renderable MaterialX elements ({}); provide node_name.", names);
}

mx::ConstDocumentPtr read_libraries_if_present(const std::filesystem::path& root, const mx::FilePathVec& folders)
{
    if (!std::filesystem::exists(root))
        return {};
    mx::DocumentPtr library = mx::createDocument();
    return mx::loadLibraries(folders, mx::FileSearchPath(root.string()), library).empty() ? nullptr : library;
}

bool load_libraries_if_present(
    const mx::DocumentPtr& document,
    const std::filesystem::path& root,
    const mx::FilePathVec& folders,
    std::unordered_set<const mx::NodeGraph*>& builtin_node_graphs,
    const mx::ConstDocumentPtr& cached_library = {}
)
{
    if (!std::filesystem::exists(root))
        return false;

    // MaterialX imports private copies from library documents into the
    // authored document. Record the identities of only those copies that the
    // import introduced: authored overrides and user XIncludes were already
    // present and therefore remain user-owned.
    std::unordered_set<const mx::NodeGraph*> graphs_before;
    for (const mx::NodeGraphPtr& graph : document->getNodeGraphs())
        graphs_before.insert(graph.get());

    bool loaded;
    if (cached_library) {
        document->importLibrary(cached_library);
        loaded = true;
    } else {
        loaded = !mx::loadLibraries(folders, mx::FileSearchPath(root.string()), document).empty();
    }
    for (const mx::NodeGraphPtr& graph : document->getNodeGraphs()) {
        if (!graphs_before.contains(graph.get()))
            builtin_node_graphs.insert(graph.get());
    }
    return loaded;
}

// Attached libraries have different lookup precedence from imported copies.
// Keep custom definitions, includes and document-wide context on the private path.
bool can_share_libraries(
    const mx::DocumentPtr& document,
    const CodeGenDesc& desc,
    const mx::ConstDocumentPtr& standard,
    const mx::ConstDocumentPtr& falcor
)
{
    for (const std::string& attribute : document->getAttributeNames()) {
        if (attribute != "version")
            return false;
    }
    for (const mx::ElementPtr& child : document->getChildren()) {
        if (!child->isA<mx::Node>() && !child->isA<mx::Output>())
            return false;
        // Even a differently typed authored element shadows an imported child.
        if (standard->getChild(child->getName()) || falcor->getChild(child->getName()))
            return false;
    }
    for (const mx::ElementPtr& element : document->traverseTree()) {
        if (element->hasSourceUri() && element->getSourceUri() != document->getSourceUri())
            return false;
        if (element->hasNamespace() || element->hasAttribute("inherit"))
            return false;
    }
    return desc.node_name.empty() || document->getDescendant(desc.node_name) != nullptr;
}

std::unordered_set<const mx::NodeGraph*>
import_libraries(const mx::DocumentPtr& document, const CodeGenDesc& desc, bool has_includes)
{
    struct BuiltinLibraries {
        mx::ConstDocumentPtr standard;
        mx::ConstDocumentPtr falcor;
    };
    const auto read_builtin_libraries = []
    {
        return BuiltinLibraries{
            read_libraries_if_present(platform::runtime_directory() / "mtlx/standard", {"libraries", "."}),
            read_libraries_if_present(platform::runtime_directory() / "mtlx/libraries", {"."}),
        };
    };
    BuiltinLibraries libraries;
    // Shipped runtime libraries are immutable for the process lifetime. Do not
    // freeze environment overrides or loadLibraries' relative-folder fallback.
    if (mx::getEnvironmentPath().isEmpty() && !std::filesystem::exists("libraries")) {
        static const BuiltinLibraries cached = read_builtin_libraries();
        libraries = cached;
    }

    std::unordered_set<const mx::NodeGraph*> builtin_node_graphs;
    if (!has_includes && libraries.standard && libraries.falcor
        && can_share_libraries(document, desc, libraries.standard, libraries.falcor)) {
        const auto prepare = [&](bool derivatives) -> mx::ConstDocumentPtr
        {
            mx::DocumentPtr library = mx::createDocument();
            library->importLibrary(libraries.standard);
            library->importLibrary(libraries.falcor);
            make_ports_compatible(library);
            if (derivatives)
                library->removeImplementation("IM_heighttonormal_vector3_genslangpt");
            document_passes::materialize_default_geomprops(*library, nullptr, {});
            std::string message;
            FALCOR_CHECK(library->validate(&message), "Prepared MaterialX library validation failed:\n{}", message);
            return library;
        };
        const auto prepared = [&]() -> mx::ConstDocumentPtr
        {
            if (desc.use_slang_derivatives) {
                static const mx::ConstDocumentPtr library = prepare(true);
                return library;
            }
            static const mx::ConstDocumentPtr library = prepare(false);
            return library;
        }();
        // Validation and mutation traverse authored children only; definitions
        // and target implementations resolve through the prepared library.
        document->setDataLibrary(prepared);
        for (const mx::NodeGraphPtr& graph : prepared->getNodeGraphs())
            builtin_node_graphs.insert(graph.get());
        return builtin_node_graphs;
    }
    bool standard_loaded = load_libraries_if_present(
        document,
        platform::runtime_directory() / "mtlx/standard",
        {"libraries", "."},
        builtin_node_graphs,
        libraries.standard
    );
    if (!standard_loaded) {
        standard_loaded |= load_libraries_if_present(
            document,
            platform::project_directory() / "external" / "MaterialX",
            {"libraries"},
            builtin_node_graphs
        );
    }
    load_libraries_if_present(
        document,
        platform::runtime_directory() / "mtlx/libraries",
        {"."},
        builtin_node_graphs,
        libraries.falcor
    );
    return builtin_node_graphs;
}

void append_resolver_paths(mx::FileSearchPath& search_path, const CodeGenDesc& desc)
{
    if (!desc.asset_resolver)
        return;
    for (const std::filesystem::path& root : desc.asset_resolver->get_search_paths())
        search_path.append(mx::FilePath(root.generic_string()));
}

std::string sanitize_shader_name(std::string value)
{
    for (char& character : value) {
        if (!std::isalnum(static_cast<unsigned char>(character)) && character != '_')
            character = '_';
    }
    if (value.empty() || std::isdigit(static_cast<unsigned char>(value.front())))
        value.insert(value.begin(), '_');
    return value;
}

} // namespace

mx::DocumentPtr
load_and_validate_document(const CodeGenDesc& desc, std::unordered_set<const mx::NodeGraph*>& builtin_node_graphs)
{
    mx::DocumentPtr document = mx::createDocument();

    // Even an empty include is an external dependency. Source URIs on imported
    // children alone cannot tell us that this document used one.
    bool has_includes = false;
    mx::XmlReadOptions read_options;
    read_options.readXIncludeFunction = [&](mx::DocumentPtr included,
                                            const mx::FilePath& file,
                                            const mx::FileSearchPath& paths,
                                            const mx::XmlReadOptions* options)
    {
        has_includes = true;
        mx::readFromXmlFile(included, file, paths, options);
    };

    // Read the authored source. A file's own directory precedes resolver roots
    // so relative includes remain local to the document that contains them.
    if (const auto* source = std::get_if<std::string>(&desc.document)) {
        mx::FileSearchPath search_path;
        append_resolver_paths(search_path, desc);
        mx::readFromXmlString(document, *source, search_path, &read_options);
    } else {
        const std::filesystem::path authored_path = std::get<std::filesystem::path>(desc.document);
        std::filesystem::path resolved_path;
        if (desc.asset_resolver)
            resolved_path = desc.asset_resolver->resolve_path(authored_path);
        else
            resolved_path = std::filesystem::absolute(authored_path).lexically_normal();
        FALCOR_CHECK(
            !resolved_path.empty() && std::filesystem::exists(resolved_path),
            "MaterialX file `{}` does not exist.",
            authored_path.string()
        );
        mx::FileSearchPath search_path(mx::FilePath(resolved_path.parent_path().generic_string()));
        append_resolver_paths(search_path, desc);
        mx::readFromXmlFile(document, mx::FilePath(resolved_path.generic_string()), search_path, &read_options);
    }

    // User documents and their expanded includes may define graphs, but shader
    // source belongs exclusively to our libraries (including custom_nodes.mtlx).
    // Reject declarations even if unused or for another target. Check before
    // library import so no source provenance or cache-invalidation tracking is needed.
    for (const mx::ImplementationPtr& implementation : document->getImplementations()) {
        FALCOR_CHECK(
            !implementation->hasAttribute("sourcecode") && !implementation->hasFile(),
            "MaterialX source implementation '{}' is unsupported in material documents. "
            "Shader source must be provided by the MaterialX library; material documents and their includes "
            "may use library nodes and custom nodegraphs.",
            implementation->getName()
        );
    }

    // Import definitions before inserting compatible-width conversions, then
    // validate the complete mutated document before selecting a root.
    builtin_node_graphs = import_libraries(document, desc, has_includes);
    make_ports_compatible(document);
    std::string validation_message;
    FALCOR_CHECK(
        document->validate(&validation_message),
        "MaterialX 1.39 validation failed with message:\n{}",
        validation_message
    );

    return document;
}

mx::DocumentPtr load_and_validate_document(const CodeGenDesc& desc)
{
    std::unordered_set<const mx::NodeGraph*> ignored_builtin_node_graphs;
    return load_and_validate_document(desc, ignored_builtin_node_graphs);
}

LoadedDocument select_document_root(mx::DocumentPtr document, const CodeGenDesc& desc)
{
    FALCOR_CHECK(document, "Cannot select a MaterialX root from a null document.");

    // Keep the selected instance: replacing it with its implementation graph
    // loses instance values and incoming connections. Let ShaderGraph::create
    // resolve those bindings, as Graph Editor's GlslMaterial::generateShader does.
    // https://github.com/AcademySoftwareFoundation/MaterialX/blob/7b64921ef1d42f2d57871e9d2c43dc11f041f26b/source/MaterialXRenderGlsl/GlslMaterial.cpp#L59
    // Additional outputs never change ordinary primary-root discovery.
    mx::ElementPtr root_element = find_root_element(document, desc);
    root_element = wrap_previewable_root_output(document, root_element);
    mx::TypedElementPtr root = root_element ? root_element->asA<mx::TypedElement>() : nullptr;
    FALCOR_CHECK(root, "MaterialX selected root is not a typed element.");

    const std::string authored_name = desc.node_name.empty() ? root->getNamePath() : desc.node_name;
    LoadedDocument result{
        std::move(document),
        std::move(root),
        {},
        {},
        desc.canonicalize_names ? "material_canonical" : sanitize_shader_name(authored_name),
        {},
    };

    const mx::NodePtr root_node = result.root->asA<mx::Node>();
    const mx::NodeDefPtr root_node_def = root_node ? root_node->getNodeDef() : nullptr;

    // Resolve exact names first so typos and attempts to select another closure
    // remain errors. Patterns may match nothing and never select closures.
    const auto is_numeric = [](const mx::OutputPtr& output)
    {
        const std::string& type = output->getType();
        return CHANNEL_COUNTS.contains(type) || type == "boolean" || type == "integer" || type == "matrix33"
            || type == "matrix44";
    };
    for (const std::string& pattern : desc.additional_output_names) {
        if (pattern.find('*') != std::string::npos)
            continue;
        mx::OutputPtr output = result.document->getOutput(pattern);
        if (root_node_def && (!output || !is_numeric(output))) {
            const mx::OutputPtr compound_output = root_node_def->getActiveOutput(pattern);
            if (compound_output && (!output || is_numeric(compound_output)))
                output = compound_output;
        }
        FALCOR_CHECK(output, "MTLX could not find additional output '{}'.", pattern);
        FALCOR_CHECK(output != result.root, "MTLX output '{}' was selected more than once.", pattern);
        FALCOR_CHECK(
            is_numeric(output),
            "MTLX additional output '{}' has unsupported type '{}'; only the main output may be material or "
            "closure-shaped.",
            pattern,
            output->getType()
        );
    }

    std::unordered_set<std::string> candidate_names;
    const auto consider_candidate = [&](const mx::OutputPtr& output, bool independent_output)
    {
        if (output == result.root || !is_numeric(output)
            || !std::any_of(
                desc.additional_output_names.begin(),
                desc.additional_output_names.end(),
                [&](const std::string& pattern)
                {
                    return matches_output_pattern(output->getName(), pattern);
                }
            ))
            return;

        const std::string& name = output->getName();
        FALCOR_CHECK(
            candidate_names.insert(name).second,
            "MTLX material AOV candidate '{}' was declared more than once.",
            name
        );
        if (desc.material_aov_registry) {
            // Runtime transport supports float1-4. Direct codegen also supports
            // ordinary numeric values such as matrices without registry IDs.
            const auto component_count_it = CHANNEL_COUNTS.find(output->getType());
            if (component_count_it == CHANNEL_COUNTS.end())
                return;
            const uint32_t component_count = sgl::narrow_cast<uint32_t>(component_count_it->second);
            const uint32_t id
                = desc.material_aov_registry->register_output(name, sgl::DataType::float32, component_count);
            result.material_aov_outputs.push_back({id, name, sgl::DataType::float32, component_count});
        }
        if (independent_output)
            result.additional_outputs.push_back(output);
    };

    if (root_node_def) {
        for (const mx::OutputPtr& output : root_node_def->getActiveOutputs())
            consider_candidate(output, false);
    }
    for (const mx::OutputPtr& output : result.document->getOutputs())
        consider_candidate(output, true);

    return result;
}

LoadedDocument load_and_select_document(const CodeGenDesc& desc)
{
    std::unordered_set<const mx::NodeGraph*> builtin_node_graphs;
    LoadedDocument loaded = select_document_root(load_and_validate_document(desc, builtin_node_graphs), desc);
    loaded.builtin_node_graphs = std::move(builtin_node_graphs);
    return loaded;
}

std::vector<RenderableElement> discover_renderable_elements(const CodeGenDesc& desc)
{
    const mx::DocumentPtr document = load_and_validate_document(desc);
    std::vector<RenderableElement> elements;
    for (const mx::TypedElementPtr& element : mx::findRenderableElements(document))
        elements.push_back(RenderableElement{element->getNamePath(), element->getType()});
    return elements;
}

} // namespace falcor::mtlx::structured
