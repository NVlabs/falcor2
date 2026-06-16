// SPDX-License-Identifier: Apache-2.0

#include "call_site_index.h"

#include <MaterialXGenShader/ShaderNodeImpl.h>

#include <unordered_set>

namespace falcor {
namespace materialx {
namespace mx139 {

namespace mx = MaterialX;

const mx::ShaderInput* CallSite::caller_input(const std::string& socket_name) const
{
    if (!caller_node)
        return nullptr;
    // ShaderNode::getInput returns a non-const pointer; const-cast for read-only API.
    return const_cast<mx::ShaderNode*>(caller_node)->getInput(socket_name);
}

const std::vector<CallSite>& CallSiteIndex::call_sites_of(const mx::ShaderGraph* callee) const
{
    static const std::vector<CallSite> k_empty;
    if (!callee)
        return k_empty;
    auto it = m_sites_by_callee.find(callee);
    if (it == m_sites_by_callee.end())
        return k_empty;
    return it->second;
}

CallSiteIndex build_call_site_index(const mx::ShaderGraph& top)
{
    CallSiteIndex index;

    std::unordered_set<const mx::ShaderGraph*> visited;
    std::vector<const mx::ShaderGraph*> work;
    work.push_back(&top);
    visited.insert(&top);

    // BFS over reachable graphs; record reachable_graphs in walk order, with
    // top at index 0.
    while (!work.empty()) {
        const mx::ShaderGraph* g = work.front();
        work.erase(work.begin());
        index.m_reachable_graphs.push_back(g);
        index.m_reachable_node_count += g->getNodes().size();

        for (const mx::ShaderNode* node : g->getNodes()) {
            mx::ShaderGraph* callee = const_cast<mx::ShaderNodeImpl&>(node->getImplementation()).getGraph();
            if (!callee)
                continue;

            CallSite site;
            site.caller_node = node;
            site.caller_subgraph = g;
            site.callee_subgraph = callee;
            index.m_sites_by_callee[callee].push_back(site);

            if (visited.insert(callee).second)
                work.push_back(callee);
        }
    }

    return index;
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
