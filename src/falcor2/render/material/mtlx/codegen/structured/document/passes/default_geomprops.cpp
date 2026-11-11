// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "default_geomprops.h"
#include "../core_reachability.h"

#include <MaterialXCore/Geom.h>
#include <MaterialXCore/Interface.h>
#include <MaterialXCore/Node.h>
#include <MaterialXCore/Types.h>

#include <utility>
#include <vector>

namespace falcor::mtlx::structured::document_passes {

namespace mx = MaterialX;

namespace {

bool has_graph_binding(const mx::PortElementPtr& port)
{
    return port
        && (port->hasNodeName() || port->hasNodeGraphString() || port->hasOutputString() || port->hasInterfaceName());
}

bool materialize_default_geomprop(const mx::GraphElementPtr& graph, const mx::NodePtr& node, const mx::InputPtr& def)
{
    if (!graph || !node || !def || has_graph_binding(node->getInput(def->getName())))
        return false;
    mx::GeomPropDefPtr geomprop = def->getDefaultGeomProp();
    if (!geomprop || !geomprop->hasGeomProp())
        return false;

    mx::NodePtr geom_node = graph->addNode(
        geomprop->getGeomProp(),
        graph->createValidChildName("__mtlx_static_pruning_defaultgeom_" + node->getName() + "_" + def->getName()),
        def->getType()
    );
    if (geomprop->hasSpace())
        geom_node->addInput("space", mx::STRING_TYPE_STRING)->setValueString(geomprop->getSpace());
    if (geomprop->hasIndex())
        geom_node->addInput("index", "integer")->setValueString(geomprop->getIndex());

    mx::InputPtr input = node->getInput(def->getName());
    if (!input)
        input = node->addInput(def->getName(), def->getType());
    input->removeAttribute(mx::PortElement::NODE_NAME_ATTRIBUTE);
    input->removeAttribute(mx::PortElement::NODE_GRAPH_ATTRIBUTE);
    input->removeAttribute(mx::PortElement::OUTPUT_ATTRIBUTE);
    input->removeAttribute(mx::ValueElement::INTERFACE_NAME_ATTRIBUTE);
    input->removeAttribute(mx::ValueElement::VALUE_ATTRIBUTE);
    input->setNodeName(geom_node->getName());
    return true;
}

} // namespace

void materialize_default_geomprops(
    mx::Document& document,
    const mx::TypedElementPtr& root,
    std::span<const mx::OutputPtr> additional_outputs
)
{
    std::unordered_set<mx::NodePtr> reachable;
    const auto add_reachable = [&](const mx::ElementPtr& element)
    {
        for (const mx::NodePtr& node : collect_reachable_core_nodes(element)) {
            if (node->getDocument().get() != &document)
                continue;
            reachable.insert(node);
        }
    };
    add_reachable(root);
    for (const mx::OutputPtr& output : additional_outputs)
        add_reachable(output);

    std::vector<mx::GraphElementPtr> graphs = {document.getSelf()->asA<mx::GraphElement>()};
    // getNodeGraphs() also includes the attached immutable library.
    for (const mx::ElementPtr& child : document.getChildren()) {
        if (const mx::NodeGraphPtr node_graph = child->asA<mx::NodeGraph>())
            graphs.push_back(node_graph);
    }
    // Graph edits invalidate MaterialX's document-wide lookup cache. Resolve all
    // definitions before mutating any graph so each lookup can reuse that cache.
    std::vector<std::pair<mx::NodePtr, mx::NodeDefPtr>> node_definitions;
    for (const mx::GraphElementPtr& graph : graphs) {
        if (!graph)
            continue;
        for (const mx::NodePtr& node : graph->getNodes()) {
            if (root && !reachable.contains(node))
                continue;
            mx::NodeDefPtr node_def = node->getNodeDef();
            if (node_def)
                node_definitions.emplace_back(node, std::move(node_def));
        }
    }
    for (const auto& [node, node_def] : node_definitions) {
        const mx::GraphElementPtr graph = node->getParent()->asA<mx::GraphElement>();
        for (const mx::InputPtr& input : node_def->getActiveInputs())
            materialize_default_geomprop(graph, node, input);
    }
}

} // namespace falcor::mtlx::structured::document_passes
