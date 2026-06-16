// SPDX-License-Identifier: Apache-2.0

#include "default_geomprop_materialization.h"

#include "core_reachability.h"

#include <MaterialXCore/Element.h>
#include <MaterialXCore/Geom.h>
#include <MaterialXCore/Interface.h>
#include <MaterialXCore/Node.h>
#include <MaterialXCore/Types.h>

#include <optional>
#include <string>
#include <vector>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace graph_prepare {

namespace mx = MaterialX;

namespace {

mx::GraphElementPtr graph_from_root(mx::DocumentPtr doc, mx::ElementPtr root_element)
{
    if (root_element) {
        if (auto graph = root_element->asA<mx::GraphElement>())
            return graph;
        if (auto parent = root_element->getParent())
            return parent->asA<mx::GraphElement>();
    }
    return doc ? doc->asA<mx::GraphElement>() : nullptr;
}

bool is_closure_type(const std::string& type)
{
    return type == mx::BSDF_TYPE_STRING || type == mx::EDF_TYPE_STRING;
}

bool is_connected_port(const mx::PortElementPtr& port)
{
    return port
        && (port->hasNodeName() || port->hasNodeGraphString() || port->hasOutputString() || port->hasInterfaceName());
}

bool port_has_explicit_binding(const mx::PortElementPtr& port)
{
    return port && (is_connected_port(port) || port->hasValueString());
}

bool port_has_graph_binding(const mx::PortElementPtr& port)
{
    return port
        && (port->hasNodeName() || port->hasNodeGraphString() || port->hasOutputString() || port->hasInterfaceName());
}

void remove_unbound_closure_input(const mx::PortElementPtr& port)
{
    mx::InputPtr input = port ? port->asA<mx::Input>() : nullptr;
    if (!input || !is_closure_type(input->getType()) || port_has_explicit_binding(input))
        return;

    mx::ElementPtr parent = input->getParent();
    if (!parent)
        return;

    if (mx::InterfaceElementPtr interface = parent->asA<mx::InterfaceElement>())
        interface->removeInput(input->getName());
    else
        parent->removeChild(input->getName());
}

void clear_port_source(const mx::PortElementPtr& port)
{
    if (!port)
        return;
    port->removeAttribute(mx::PortElement::NODE_NAME_ATTRIBUTE);
    port->removeAttribute(mx::PortElement::NODE_GRAPH_ATTRIBUTE);
    port->removeAttribute(mx::PortElement::OUTPUT_ATTRIBUTE);
    port->removeAttribute(mx::ValueElement::INTERFACE_NAME_ATTRIBUTE);
    port->removeAttribute(mx::ValueElement::VALUE_ATTRIBUTE);
    remove_unbound_closure_input(port);
}

bool materialize_default_geomprop_input_with_spec(
    const mx::GraphElementPtr& graph,
    const mx::NodePtr& node,
    const std::string& input_name,
    const std::string& input_type,
    const std::string& geomprop_name,
    const std::optional<std::string>& geomprop_space,
    const std::optional<std::string>& geomprop_index
)
{
    if (!graph || !node || geomprop_name.empty())
        return false;

    mx::InputPtr input = node->getInput(input_name);
    if (port_has_graph_binding(input))
        return false;

    const std::string stem = "__mx139_static_pruning_defaultgeom_" + node->getName() + "_" + input_name;
    mx::NodePtr geom_node = graph->addNode(geomprop_name, graph->createValidChildName(stem), input_type);
    if (!geom_node)
        return false;

    if (geomprop_space)
        geom_node->addInput("space", mx::STRING_TYPE_STRING)->setValueString(*geomprop_space);
    if (geomprop_index)
        geom_node->addInput("index", "integer")->setValueString(*geomprop_index);

    if (!input)
        input = node->addInput(input_name, input_type);

    clear_port_source(input);
    input->setNodeName(geom_node->getName());
    return true;
}

bool materialize_default_geomprop_input(
    const mx::GraphElementPtr& graph,
    const mx::NodePtr& node,
    const mx::InputPtr& node_def_input
)
{
    if (!node_def_input)
        return false;
    mx::GeomPropDefPtr geomprop = node_def_input->getDefaultGeomProp();
    if (!geomprop || !geomprop->hasGeomProp())
        return false;
    return materialize_default_geomprop_input_with_spec(
        graph,
        node,
        node_def_input->getName(),
        node_def_input->getType(),
        geomprop->getGeomProp(),
        geomprop->hasSpace() ? std::optional<std::string>(geomprop->getSpace()) : std::nullopt,
        geomprop->hasIndex() ? std::optional<std::string>(geomprop->getIndex()) : std::nullopt
    );
}

uint32_t materialize_default_geomprop_inputs(const mx::NodePtr& node)
{
    if (!node)
        return 0;

    mx::GraphElementPtr graph = node->getParent() ? node->getParent()->asA<mx::GraphElement>() : nullptr;
    mx::NodeDefPtr node_def = node->getNodeDef();
    if (!graph || !node_def)
        return 0;

    uint32_t count = 0;
    for (mx::InputPtr node_def_input : node_def->getActiveInputs())
        if (materialize_default_geomprop_input(graph, node, node_def_input))
            ++count;
    return count;
}

} // namespace

uint32_t materialize_reachable_default_geomprop_inputs(mx::DocumentPtr doc, const mx::ElementPtr& root_element)
{
    mx::GraphElementPtr graph = graph_from_root(doc, root_element);
    if (!graph || !root_element)
        return 0;

    uint32_t count = 0;
    const auto reachable_core_nodes = collect_reachable_core_nodes(root_element);
    const std::vector<mx::NodePtr> nodes = graph->getNodes();
    for (const mx::NodePtr& node : nodes) {
        if (reachable_core_nodes.contains(node))
            count += materialize_default_geomprop_inputs(node);
    }
    return count;
}

} // namespace graph_prepare
} // namespace mx139
} // namespace materialx
} // namespace falcor
