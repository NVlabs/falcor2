// SPDX-License-Identifier: Apache-2.0

#include "core_reachability.h"

#include <MaterialXCore/Interface.h>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace graph_prepare {

namespace mx = MaterialX;

namespace {

void visit_upstream_core_port(
    const mx::PortElementPtr& port,
    std::unordered_set<mx::NodePtr>& reachable,
    std::unordered_set<mx::PortElementPtr>& visited_ports
);

void visit_upstream_core_node(const mx::NodePtr& node, std::unordered_set<mx::NodePtr>& reachable)
{
    if (!node || !reachable.insert(node).second)
        return;

    std::unordered_set<mx::PortElementPtr> visited_ports;
    for (mx::InputPtr input : node->getInputs())
        visit_upstream_core_port(input, reachable, visited_ports);
}

void visit_upstream_core_port(
    const mx::PortElementPtr& port,
    std::unordered_set<mx::NodePtr>& reachable,
    std::unordered_set<mx::PortElementPtr>& visited_ports
)
{
    if (!port || !visited_ports.insert(port).second)
        return;

    if (mx::NodePtr node = port->getConnectedNode()) {
        visit_upstream_core_node(node, reachable);
        return;
    }

    if (mx::OutputPtr output = port->getConnectedOutput())
        visit_upstream_core_port(output, reachable, visited_ports);
}

} // namespace

std::unordered_set<mx::NodePtr> collect_reachable_core_nodes(const mx::ElementPtr& root_element)
{
    std::unordered_set<mx::NodePtr> reachable;
    if (!root_element)
        return reachable;

    if (mx::NodePtr node = root_element->asA<mx::Node>()) {
        visit_upstream_core_node(node, reachable);
        return reachable;
    }

    std::unordered_set<mx::PortElementPtr> visited_ports;
    if (mx::PortElementPtr port = root_element->asA<mx::PortElement>())
        visit_upstream_core_port(port, reachable, visited_ports);
    return reachable;
}

} // namespace graph_prepare
} // namespace mx139
} // namespace materialx
} // namespace falcor
