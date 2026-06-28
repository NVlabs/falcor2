// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderNode.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace falcor {
namespace materialx {
namespace mx139 {

// A single instantiation of a shared inner ShaderGraph at one call site.
//
// `caller_node` is the CompoundNode-backed ShaderNode in `caller_subgraph`
// whose `getImplementation().getGraph()` returns `callee_subgraph`.
struct CallSite {
    const MaterialX::ShaderNode* caller_node = nullptr;
    const MaterialX::ShaderGraph* caller_subgraph = nullptr;
    const MaterialX::ShaderGraph* callee_subgraph = nullptr;

    // Convenience: caller's ShaderInput driving the named callee socket
    // (i.e. the ShaderInput on `caller_node` named `socket_name`). Returns
    // nullptr if the caller has no such input.
    const MaterialX::ShaderInput* caller_input(const std::string& socket_name) const;
};

class CallSiteIndex {
public:
    // All call sites whose callee is `callee`. Returns an empty vector if
    // `callee` is unreachable from the top graph or `callee == nullptr`.
    const std::vector<CallSite>& call_sites_of(const MaterialX::ShaderGraph* callee) const;

    // All reachable subgraphs in deterministic walk order. The top graph is
    // index 0.
    const std::vector<const MaterialX::ShaderGraph*>& reachable_graphs() const { return m_reachable_graphs; }

    size_t reachable_graph_count() const { return m_reachable_graphs.size(); }
    size_t reachable_node_count() const { return m_reachable_node_count; }

private:
    friend CallSiteIndex build_call_site_index(const MaterialX::ShaderGraph& top);

    std::vector<const MaterialX::ShaderGraph*> m_reachable_graphs;
    std::unordered_map<const MaterialX::ShaderGraph*, std::vector<CallSite>> m_sites_by_callee;
    size_t m_reachable_node_count = 0;
};

// Walk the top-level graph and every nested ShaderGraph reachable through
// `node->getImplementation().getGraph()`. The walk is deterministic and
// records each CompoundNode instantiation.
CallSiteIndex build_call_site_index(const MaterialX::ShaderGraph& top);

} // namespace mx139
} // namespace materialx
} // namespace falcor
