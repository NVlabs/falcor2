// SPDX-License-Identifier: Apache-2.0

#include "closure_simplification.h"

#include "../static_input_query/operator_match.h"

#include "falcor2/core/error.h"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Types.h>
#include <MaterialXCore/Value.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderNodeImpl.h>
#include <MaterialXGenShader/TypeDesc.h>

#include <atomic>
#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {

namespace mx = MaterialX;

namespace {

// ShaderGraph::createNode overwrites _nodeMap on name collision and drops
// the previous shared_ptr ref, which would free a node still referenced
// from _nodeOrder and from live ShaderInput connections. Probe the graph
// and bump the counter until the name is unused.
std::atomic<uint64_t> g_synth_counter{0};

std::string fresh_name(const mx::ShaderGraph& graph, const char* prefix)
{
    while (true) {
        std::string candidate_name = std::string("_closure_prune_") + prefix + "_"
            + std::to_string(g_synth_counter.fetch_add(1, std::memory_order_relaxed));
        if (!graph.getNode(candidate_name))
            return candidate_name;
    }
}

// Pick the right multiply nodedef for a closure output type. Returns
// nullptr if the type is not a recognised surface closure.
mx::ConstNodeDefPtr multiply_nodedef_for(const mx::ShaderGraph& graph, const mx::TypeDesc& closure_type)
{
    mx::ConstDocumentPtr doc = graph.getDocument();
    if (!doc)
        return nullptr;
    const std::string name = closure_type.getName();
    if (name == "BSDF")
        return doc->getNodeDef("ND_multiply_bsdfF");
    if (name == "EDF")
        return doc->getNodeDef("ND_multiply_edfF");
    return nullptr;
}

mx::ConstNodeDefPtr invert_float_nodedef(const mx::ShaderGraph& graph)
{
    mx::ConstDocumentPtr doc = graph.getDocument();
    return doc ? doc->getNodeDef("ND_invert_float") : nullptr;
}

// Connect `dst` to whatever source feeds `src_input` (either an upstream
// output, or a literal value lifted onto dst). Caller guarantees src_input
// has either a connection or a value; classify never emits a directive on
// a mix whose weight socket is bare.
void wire_through(mx::ShaderInput* dst, const mx::ShaderInput* src_input)
{
    FALCOR_ASSERT(dst && src_input);
    if (mx::ShaderOutput* up = const_cast<mx::ShaderOutput*>(src_input->getConnection())) {
        dst->makeConnection(up);
        return;
    }
    mx::ValuePtr v = src_input->getValue();
    FALCOR_ASSERT(v);
    dst->setValue(v);
}

// Apply the rewrite for an asymmetric `mix` (one branch KnownEmpty, other kept).
// Builds `multiply(live, w)` for fg-live or `multiply(live, invert_float(w))`
// for bg-live and replaces `mix.out` with the new multiply output. This is one
// output rewrite and also one asymmetric-mix rewrite, so
// asymmetric_mix_rewrites is a subset of output_rewrites.
bool rewrite_asymmetric_mix(
    mx::ShaderGraph& graph,
    mx::GenContext& context,
    mx::ShaderNode& mix_node,
    const AsymmetricMixFoldPlan& plan,
    graph_prepare::ClosurePruningCounters& out
)
{
    mx::ShaderOutput* mix_out = mix_node.getOutput(0);
    if (!mix_out)
        return false;

    mx::ConstNodeDefPtr mul_def = multiply_nodedef_for(graph, mix_out->getType());
    if (!mul_def)
        return false;

    const bool keep_fg = plan.kept_side == AsymmetricMixFoldPlan::KeptSide::Fg;
    mx::ShaderInput* kept_input = mix_node.getInput(keep_fg ? "fg" : "bg");
    const mx::ShaderInput* weight_input = mix_node.getInput("mix");
    if (!kept_input || !weight_input)
        return false;
    mx::ShaderOutput* kept_upstream = kept_input->getConnection();
    if (!kept_upstream)
        return false; // Kept side must be connected; classifier should have caught this.

    const std::string mul_name = fresh_name(graph, "mul");
    mx::ShaderNode* mul = graph.createNode(mul_name, mul_name, mul_def, context);
    FALCOR_CHECK(
        mul != nullptr,
        "Failed to create asymmetric mix helper '{}' in graph '{}'.",
        mul_name,
        graph.getName()
    );
    mx::ShaderInput* mul_in1 = mul->getInput("in1");
    mx::ShaderInput* mul_in2 = mul->getInput("in2");
    FALCOR_CHECK(
        mul_in1 && mul_in2,
        "Asymmetric mix helper '{}' in graph '{}' is missing multiply inputs.",
        mul_name,
        graph.getName()
    );
    mul_in1->makeConnection(kept_upstream);

    if (keep_fg) {
        wire_through(mul_in2, weight_input);
    } else {
        mx::ConstNodeDefPtr inv_def = invert_float_nodedef(graph);
        if (!inv_def)
            return false;
        const std::string inv_name = fresh_name(graph, "inv");
        mx::ShaderNode* inv = graph.createNode(inv_name, inv_name, inv_def, context);
        FALCOR_CHECK(
            inv != nullptr,
            "Failed to create asymmetric mix invert helper '{}' in graph '{}'.",
            inv_name,
            graph.getName()
        );
        mx::ShaderInput* inv_in = inv->getInput("in");
        mx::ShaderOutput* inv_out = inv->getOutput(0);
        FALCOR_CHECK(
            inv_in && inv_out,
            "Asymmetric mix invert helper '{}' in graph '{}' is missing input or output.",
            inv_name,
            graph.getName()
        );
        wire_through(inv_in, weight_input);
        mul_in2->makeConnection(inv_out);
    }

    graph.replaceOutput(mix_out, mul->getOutput(0));
    ++out.output_rewrites;
    ++out.asymmetric_mix_rewrites;
    return true;
}

// Apply RewireToInput(input_name) by rewiring all consumers of `node.out` to
// whatever upstream feeds `input_name`. This is one output rewrite regardless
// of how many consumer inputs MaterialX redirects internally. Bare input
// sockets (no upstream) are left alone; top-shader collapse is detected by the
// driver via the classification map.
bool rewrite_to_input(
    mx::ShaderGraph& graph,
    mx::ShaderNode& node,
    const std::string& input_name,
    graph_prepare::ClosurePruningCounters& out
)
{
    mx::ShaderOutput* node_out = node.getOutput(0);
    mx::ShaderInput* src = node.getInput(input_name);
    if (!node_out || !src)
        return false;
    mx::ShaderOutput* upstream = src->getConnection();
    if (!upstream)
        return false;
    graph.replaceOutput(node_out, upstream);
    ++out.output_rewrites;
    return true;
}

// Advanced codegen rejects graphs where the same closure output is reachable
// via more than one live path. A RewireToInput rewrite on a weighted_layer
// replaces wl.out with the kept input's upstream, then prunes wl; the
// upstream ends up with (existing_consumers - 1 for wl) + wl.out_consumers
// live refs. Skip if that sum >= 2 -- a later iteration may collapse one
// side. Applies to either `top` or `base`.
bool would_duplicate_upstream(const mx::ShaderNode& wl_node, const std::string& input_name)
{
    const mx::ShaderInput* in = wl_node.getInput(input_name);
    const mx::ShaderOutput* up = in ? in->getConnection() : nullptr;
    if (!up)
        return false;
    const mx::ShaderOutput* out = wl_node.getOutput();
    std::size_t up_consumers = up->getConnections().size(); // includes wl's own input
    std::size_t out_consumers = out ? out->getConnections().size() : 0;
    return (up_consumers - 1) + out_consumers >= 2;
}

uint32_t prune_graph(
    mx::ShaderGraph& graph,
    mx::GenContext& context,
    const ClosurePruningAnalysisResult& cr,
    graph_prepare::ClosurePruningCounters& out
)
{
    bool mutated = false;
    // Snapshot the node list: rewrites do not insert into _nodeOrder until
    // they call createNode, which appends; we want to skip any nodes we
    // synthesize this pass.
    std::vector<mx::ShaderNode*> nodes_snapshot;
    nodes_snapshot.reserve(graph.getNodes().size());
    for (mx::ShaderNode* n : graph.getNodes())
        nodes_snapshot.push_back(n);

    for (mx::ShaderNode* node : nodes_snapshot) {
        auto it = cr.pruning.classification.find(node);
        if (it == cr.pruning.classification.end())
            continue;
        const NodeClassification& cls = it->second;
        if (cls.kind == NodeClassification::Kind::RewireToInput) {
            if (is_weighted_layer_node(*node) && would_duplicate_upstream(*node, cls.rewire_input))
                continue;
            if (rewrite_to_input(graph, *node, cls.rewire_input, out))
                mutated = true;
            continue;
        }
        if (cls.kind == NodeClassification::Kind::Keep) {
            auto dit = cr.pruning.asymmetric_mix_fold_plans.find(node);
            if (dit != cr.pruning.asymmetric_mix_fold_plans.end()) {
                if (rewrite_asymmetric_mix(graph, context, *node, dit->second, out))
                    mutated = true;
            }
            continue;
        }
        // RemoveAsEmpty: handled implicitly. Consumer rules (RewireToInput /
        // asymmetric mix) bubble KnownEmpty up the chain. KnownEmpty leaves
        // with no surviving consumer will be removed by removeUnusedNodes()
        // below.
    }

    if (!mutated)
        return 0;
    const size_t before = graph.getNodes().size();
    graph.removeUnusedNodes();
    // replaceOutput can rewire a consumer to an upstream that appears later
    // in _nodeOrder than the consumer, leaving the graph topologically
    // unsorted. MaterialX codegen emits in _nodeOrder, so re-sort.
    graph.topologicalSort();
    const size_t after = graph.getNodes().size();
    return before > after ? static_cast<uint32_t>(before - after) : 0u;
}

} // namespace

bool apply_closure_simplification(
    const ClosurePruningAnalysisResult& cr,
    const CallSiteIndex& index,
    mx::GenContext& context,
    graph_prepare::ClosurePruningCounters& out
)
{
    uint32_t pruned = 0;
    const uint32_t output_rewrites_before = out.output_rewrites;
    for (const mx::ShaderGraph* g : index.reachable_graphs())
        pruned += prune_graph(const_cast<mx::ShaderGraph&>(*g), context, cr, out);
    out.nodes_pruned += pruned;
    return pruned > 0 || out.output_rewrites != output_rewrites_before;
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
