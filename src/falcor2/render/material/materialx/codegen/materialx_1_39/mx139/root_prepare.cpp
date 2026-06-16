// SPDX-License-Identifier: Apache-2.0

#include "root_prepare.h"

#include "falcor2/core/error.h"

#include <MaterialXCore/Node.h>
#include <MaterialXCore/Types.h>
#include <MaterialXGenShader/Util.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace root_prepare {

namespace mx = MaterialX;

namespace {

// Channel tables used when the requested root is a value output rather than a
// material. They mirror the MaterialX raster preview conventions so the rewrite
// path preserves the legacy visible preview behavior.
// clang-format off
const std::unordered_map<std::string, std::vector<char>> k_preview_channel_vectors = {
    { "boolean", { 'x', 'x', 'x', 'x' } },
    { "integer", { 'x', 'x', 'x', 'x' } },
    { "float", { 'x', 'x', 'x', 'x' } },
    { "color3", { 'r', 'g', 'b', 'b' } },
    { "color4", { 'r', 'g', 'b', 'a' } },
    { "vector2", { 'x', 'y', '0', '0' } },
    { "vector3", { 'x', 'y', 'z', 'z' } },
    { "vector4", { 'x', 'y', 'z', 'w' } }
};

const std::unordered_map<std::string, size_t> k_preview_channel_counts = {
    { "boolean", 1 },
    { "integer", 1 },
    { "float", 1 },
    { "color3", 3 },
    { "color4", 4 },
    { "vector2", 2 },
    { "vector3", 3 },
    { "vector4", 4 }
};
// clang-format on

const std::unordered_set<std::string> k_black_preview_types = {
    "boolean",
    "matrix33",
    "matrix44",
};

bool can_preview_output_as_diffuse(const std::string& type)
{
    return (k_preview_channel_vectors.contains(type) && k_preview_channel_counts.contains(type))
        || k_black_preview_types.contains(type);
}

bool is_black_preview_type(const std::string& type)
{
    return k_black_preview_types.contains(type);
}

bool is_scalar_preview_type(const std::string& type)
{
    return type == "boolean" || type == "integer" || type == "float";
}

bool has_output_channels(const mx::OutputPtr& output)
{
    return output->hasAttribute("channels");
}

std::string get_output_channels(const mx::OutputPtr& output)
{
    return has_output_channels(output) ? output->getAttribute("channels") : std::string();
}

void connect_to_output(mx::InputPtr input, mx::OutputPtr output)
{
    if (auto node_graph = output->getParent()->asA<mx::NodeGraph>()) {
        if (mx::NodeDefPtr node_def = node_graph->getNodeDef()) {
            // A nodegraph with a nodedef is an implementation graph. When a preview
            // output points at such a graph, materialize a node instance so nodedef
            // defaults and version-specific implementations are resolved just like
            // regular MaterialX node use.
            mx::DocumentPtr doc = output->getDocument();
            mx::NodePtr instance = doc->addNodeInstance(
                node_def,
                doc->createValidChildName("mx_preview_" + output->getName() + "_instance")
            );
            input->setNodeName(instance->getName());
            if (node_graph->getOutputCount() > 1)
                input->setOutputString(output->getName());
            return;
        }
    }

    input->setConnectedOutput(output);
}

void connect_to_port_source(mx::InputPtr input, const mx::PortElementPtr& source_port)
{
    if (auto output = source_port->asA<mx::Output>()) {
        connect_to_output(input, output);
        return;
    }

    if (source_port->hasNodeName()) {
        input->setNodeName(source_port->getNodeName());
        if (source_port->hasOutputString())
            input->setOutputString(source_port->getOutputString());
        return;
    }

    if (source_port->hasNodeGraphString()) {
        input->setNodeGraphString(source_port->getNodeGraphString());
        if (source_port->hasOutputString())
            input->setOutputString(source_port->getOutputString());
        return;
    }
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
        if (const mx::NodeDefPtr& node_def = connected_node->getNodeDef())
            output = node_def->getOutput(output_string);
    } else if (port->hasNodeGraphString()) {
        const std::string& node_graph_name = port->getNodeGraphString();
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

std::string diffuse_preview_channels(const std::string& source_type, const std::string& output_channels)
{
    std::string channels;
    channels.reserve(3);

    if (output_channels.empty()) {
        auto src_chars_it = k_preview_channel_vectors.find(source_type);
        if (src_chars_it == k_preview_channel_vectors.end())
            return {};
        for (size_t i = 0; i < 3; ++i)
            channels.push_back(src_chars_it->second[i]);
        return channels;
    }

    for (size_t i = 0; i < 3; ++i)
        channels.push_back(output_channels[std::min(i, output_channels.size() - 1)]);
    return channels;
}

std::string separate_node_category(const std::string& source_type)
{
    auto count_it = k_preview_channel_counts.find(source_type);
    FALCOR_CHECK(
        count_it != k_preview_channel_counts.end(),
        "Cannot separate unsupported MaterialX type `{}`",
        source_type
    );
    return "separate" + std::to_string(count_it->second);
}

std::string separate_node_def(const std::string& source_type)
{
    return "ND_" + separate_node_category(source_type) + "_" + source_type;
}

std::string separated_output_name(const std::string& source_type, char channel)
{
    static const std::unordered_map<char, std::string> k_color_outputs = {
        {'r', "outr"},
        {'g', "outg"},
        {'b', "outb"},
        {'a', "outa"},
    };
    static const std::unordered_map<char, std::string> k_vector_outputs = {
        {'x', "outx"},
        {'y', "outy"},
        {'z', "outz"},
        {'w', "outw"},
    };

    const auto& outputs = source_type.starts_with("color") ? k_color_outputs : k_vector_outputs;
    auto output_it = outputs.find(channel);
    FALCOR_CHECK(
        output_it != outputs.end(),
        "MaterialX preview channel `{}` is not valid for source type `{}`",
        channel,
        source_type
    );
    return output_it->second;
}

mx::NodePtr convert_output_to_color3(mx::DocumentPtr doc, const std::string& prefix, mx::OutputPtr output)
{
    mx::NodePtr convert = doc->addNode("convert", doc->createValidChildName(prefix + "_convert_to_color3"), "color3");
    connect_to_output(convert->addInput("in", output->getType()), output);
    return convert;
}

mx::NodePtr compose_channel_preview_to_color3(mx::DocumentPtr doc, const std::string& prefix, mx::OutputPtr output)
{
    const std::string source_type = get_connected_source_type(output);
    FALCOR_CHECK(!source_type.empty(), "Cannot determine source type for MaterialX output `{}`", output->getNamePath());

    const std::string channels = diffuse_preview_channels(source_type, get_output_channels(output));
    FALCOR_CHECK(
        !channels.empty(),
        "MaterialX output `{}` has no color3 preview channels from `{}`",
        output->getNamePath(),
        source_type
    );

    mx::NodePtr separate;
    if (!is_scalar_preview_type(source_type)) {
        separate = doc->addNode(
            separate_node_category(source_type),
            doc->createValidChildName(prefix + "_separate"),
            "multioutput"
        );
        separate->setNodeDefString(separate_node_def(source_type));
        connect_to_port_source(separate->addInput("in", source_type), output);
    }

    mx::NodePtr combine = doc->addNode("combine3", doc->createValidChildName(prefix + "_combine_color3"), "color3");
    for (size_t i = 0; i < 3; ++i) {
        mx::InputPtr input = combine->addInput("in" + std::to_string(i + 1), "float");
        const char channel = channels[i];
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
    mx::InputPtr color_input,
    mx::DocumentPtr doc,
    const std::string& prefix,
    mx::OutputPtr output
)
{
    if (is_black_preview_type(output->getType())) {
        // MaterialX 1.39 raster packs unsupported non-color preview outputs, including boolean and matrices,
        // to black in SlangShaderGenerator::toVec4. Match that visible behavior instead of running
        // material lighting diagnostics on a value output.
        color_input->setValueString("0.0, 0.0, 0.0");
        return;
    }

    if (has_output_channels(output)) {
        color_input->setConnectedNode(compose_channel_preview_to_color3(doc, prefix, output));
        return;
    }

    if (output->getType() == "color3") {
        connect_to_output(color_input, output);
        return;
    }

    color_input->setConnectedNode(convert_output_to_color3(doc, prefix, output));
}

mx::NodePtr wrap_output_in_oren_nayar_preview(mx::DocumentPtr doc, mx::OutputPtr output)
{
    FALCOR_CHECK((bool)doc, "Cannot wrap MaterialX output without a document");
    FALCOR_CHECK((bool)output, "Cannot wrap null MaterialX value output");
    FALCOR_CHECK(
        can_preview_output_as_diffuse(output->getType()),
        "MaterialX output `{}` has type `{}` and cannot be previewed as an Oren-Nayar BSDF",
        output->getNamePath(),
        output->getType()
    );

    const std::string prefix = doc->createValidChildName("mx_preview_" + output->getName());

    mx::NodePtr bsdf = doc->addNode("oren_nayar_diffuse_bsdf", prefix + "_bsdf", "BSDF");
    connect_diffuse_preview_color(bsdf->addInput("color", "color3"), doc, prefix, output);
    bsdf->addInput("roughness", "float")->setValueString("0.0");

    mx::NodePtr surface = doc->addNode("surface", prefix + "_surface", mx::SURFACE_SHADER_TYPE_STRING);
    surface->setConnectedNode("bsdf", bsdf);

    mx::NodePtr material = doc->addNode("surfacematerial", prefix + "_material", mx::MATERIAL_TYPE_STRING);
    material->setConnectedNode("surfaceshader", surface);
    return material;
}

mx::ElementPtr resolve_material_implementation_root(mx::ElementPtr root_element)
{
    mx::NodePtr node = root_element ? root_element->asA<mx::Node>() : nullptr;
    if (!node || node->getType() != mx::MATERIAL_TYPE_STRING || node->getCategory() == "surfacematerial")
        return root_element;

    mx::InterfaceElementPtr implementation = node->getImplementation();
    mx::NodeGraphPtr node_graph = implementation ? implementation->asA<mx::NodeGraph>() : nullptr;
    if (!node_graph)
        return root_element;

    mx::OutputPtr output = node_graph->getOutput("out");
    if (!output) {
        for (mx::OutputPtr candidate : node_graph->getActiveOutputs()) {
            if (candidate->getType() == mx::MATERIAL_TYPE_STRING
                || candidate->getType() == mx::SURFACE_SHADER_TYPE_STRING) {
                output = candidate;
                break;
            }
        }
    }

    return output ? output : root_element;
}

mx::ElementPtr wrap_previewable_root_output(mx::DocumentPtr doc, mx::ElementPtr root_element)
{
    mx::OutputPtr output = root_element ? root_element->asA<mx::Output>() : nullptr;
    if (!output || !can_preview_output_as_diffuse(output->getType()))
        return root_element;

    return wrap_output_in_oren_nayar_preview(doc, output);
}

std::vector<mx::TypedElementPtr> find_top_level_nodes_of_type(mx::DocumentPtr doc, const std::string& type)
{
    std::vector<mx::TypedElementPtr> elements;
    for (const mx::ElementPtr& child : doc->getChildren()) {
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
    if (!typed)
        return false;

    const bool is_root_kind = element->asA<mx::Node>() || element->asA<mx::Output>();
    if (!is_root_kind)
        return false;

    const std::string type = typed->getType();
    return type == mx::MATERIAL_TYPE_STRING || type == mx::SURFACE_SHADER_TYPE_STRING;
}

mx::ElementPtr find_unique_root_element_by_local_name(mx::DocumentPtr doc, const std::string& name)
{
    std::vector<mx::ElementPtr> matches;
    for (mx::ElementPtr element : doc->traverseTree()) {
        if (element->getName() != name || !is_named_root_candidate(element))
            continue;
        matches.push_back(element);
    }

    if (matches.size() == 1)
        return matches.front();
    if (matches.empty())
        FALCOR_THROW("MX139 could not find MaterialX element '{}'.", name);

    std::string names = matches.front()->getNamePath();
    for (size_t i = 1; i < matches.size(); ++i)
        names += ", " + matches[i]->getNamePath();
    FALCOR_THROW(
        "MX139 found multiple MaterialX elements named '{}' ({}); provide a qualified name path.",
        name,
        names
    );
}

std::vector<mx::TypedElementPtr> find_unconsumed_top_level_bsdfs(mx::DocumentPtr doc)
{
    std::unordered_set<std::string> consumed_node_names;
    for (const mx::ElementPtr& child : doc->getChildren()) {
        mx::NodePtr node = child->asA<mx::Node>();
        if (!node)
            continue;
        for (const mx::InputPtr& input : node->getInputs()) {
            if (input->hasNodeName())
                consumed_node_names.insert(input->getNodeName());
        }
    }

    std::vector<mx::TypedElementPtr> elements;
    for (const mx::TypedElementPtr& typed : find_top_level_nodes_of_type(doc, mx::BSDF_TYPE_STRING)) {
        if (consumed_node_names.find(typed->getName()) == consumed_node_names.end())
            elements.push_back(typed);
    }
    return elements;
}

mx::ElementPtr find_root_element(mx::DocumentPtr doc, const CodeGenDesc& desc)
{
    if (!desc.node_name.empty()) {
        if (auto element = doc->getDescendant(desc.node_name))
            return element;
        return find_unique_root_element_by_local_name(doc, desc.node_name);
    }

    std::vector<mx::TypedElementPtr> elements = mx::findRenderableElements(doc);
    if (elements.size() == 1)
        return elements.front();
    if (elements.empty()) {
        elements = find_top_level_nodes_of_type(doc, mx::SURFACE_SHADER_TYPE_STRING);
        if (elements.size() == 1)
            return elements.front();
    }
    if (elements.empty()) {
        elements = find_unconsumed_top_level_bsdfs(doc);
        if (elements.size() == 1)
            return elements.front();
    }
    if (elements.empty())
        FALCOR_THROW("MX139 could not automatically find a renderable MaterialX element.");

    std::string names = elements.front()->getNamePath();
    for (size_t i = 1; i < elements.size(); ++i)
        names += ", " + elements[i]->getNamePath();
    FALCOR_THROW("MX139 found multiple renderable MaterialX elements ({}); provide node_name.", names);
}

} // namespace

RootPrepareResult prepare_root(mx::DocumentPtr doc, const CodeGenDesc& desc)
{
    mx::ElementPtr root_element = find_root_element(doc, desc);
    root_element = resolve_material_implementation_root(root_element);
    root_element = wrap_previewable_root_output(doc, root_element);
    return {root_element};
}

} // namespace root_prepare
} // namespace mx139
} // namespace materialx
} // namespace falcor
