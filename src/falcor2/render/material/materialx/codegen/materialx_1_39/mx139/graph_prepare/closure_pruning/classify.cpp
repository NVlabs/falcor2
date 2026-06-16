// SPDX-License-Identifier: Apache-2.0

#include "classify.h"

#include "../static_input_query/input_static_values.h"
#include "../static_input_query/operator_match.h"

#include "falcor2/core/error.h"

#include <utility>

namespace falcor {
namespace materialx {
namespace mx139 {

namespace mx = MaterialX;

namespace {

NodeClassification remove_as_empty()
{
    return {NodeClassification::Kind::RemoveAsEmpty, {}};
}

NodeClassification keep_node()
{
    return {NodeClassification::Kind::Keep, {}};
}

NodeClassification rewire_to_input(std::string input_name)
{
    return {NodeClassification::Kind::RewireToInput, std::move(input_name)};
}

struct MixClosureClassification {
    NodeClassification classification = keep_node();
    bool has_asymmetric_mix_fold_plan = false;
    AsymmetricMixFoldPlan asymmetric_mix_fold_plan;
};

// Set the per-graph input environment by populating output_static_values for
// each ShaderGraphInputSocket of `graph`. `graph` is either the top graph
// (env_for_top) or an inner graph whose env is the merge across call sites.
void seed_top_graph_env(const mx::ShaderGraph& top, OutputStaticValueMap& out)
{
    for (const mx::ShaderOutput* socket : top.getInputSockets()) {
        out[socket] = socket_default_static_value(socket);
    }
}

void seed_inner_graph_env(
    const mx::ShaderGraph& callee,
    const std::vector<CallSite>& sites,
    const OutputStaticValueMap& prev,
    OutputStaticValueMap& current
)
{
    for (const mx::ShaderOutput* socket : callee.getInputSockets()) {
        const std::string& name = socket->getName();
        OutputStaticValue acc;
        bool initialised = false;
        for (const CallSite& site : sites) {
            const mx::ShaderInput* caller_input = site.caller_input(name);
            OutputStaticValue site_val
                = caller_input ? input_static_value(*caller_input, prev) : socket_default_static_value(socket);
            acc = initialised ? merge_output_static_values(acc, site_val) : site_val;
            initialised = true;
        }
        if (!initialised) {
            // No recorded call sites (unreachable through call-site index):
            // fall back to socket default.
            acc = socket_default_static_value(socket);
        }
        current[socket] = acc;
    }
}

// Classify a single mix-closure call. Inputs are `fg`, `bg`, `mix`.
MixClosureClassification classify_mix_closure(const mx::ShaderNode& node, const OutputStaticValueMap& current)
{
    const mx::ShaderInput* fg_input = node.getInput("fg");
    const mx::ShaderInput* bg_input = node.getInput("bg");
    const mx::ShaderInput* mix_input = node.getInput("mix");
    const ClosureStaticValue foreground_closure = closure_of(fg_input, current);
    const ClosureStaticValue background_closure = closure_of(bg_input, current);
    const ScalarStaticValue mix_weight = scalar_of(mix_input, current);

    MixClosureClassification result;
    if (is_empty_closure(foreground_closure) && is_empty_closure(background_closure)) {
        result.classification = remove_as_empty();
    } else if (is_zero_static_value(mix_weight)) {
        result.classification = is_empty_closure(background_closure) ? remove_as_empty() : rewire_to_input("bg");
    } else if (is_one_static_value(mix_weight)) {
        result.classification = is_empty_closure(foreground_closure) ? remove_as_empty() : rewire_to_input("fg");
    } else if (is_empty_closure(foreground_closure) && !is_empty_closure(background_closure)) {
        result.classification = keep_node();
        result.has_asymmetric_mix_fold_plan = true;
        result.asymmetric_mix_fold_plan.kept_side = AsymmetricMixFoldPlan::KeptSide::Bg;
    } else if (!is_empty_closure(foreground_closure) && is_empty_closure(background_closure)) {
        result.classification = keep_node();
        result.has_asymmetric_mix_fold_plan = true;
        result.asymmetric_mix_fold_plan.kept_side = AsymmetricMixFoldPlan::KeptSide::Fg;
    }
    return result;
}

NodeClassification classify_add_closure(const mx::ShaderNode& node, const OutputStaticValueMap& current)
{
    const ClosureStaticValue input1_closure = closure_of(node.getInput("in1"), current);
    const ClosureStaticValue input2_closure = closure_of(node.getInput("in2"), current);
    if (is_empty_closure(input1_closure) && is_empty_closure(input2_closure))
        return remove_as_empty();
    if (is_empty_closure(input1_closure) && !is_empty_closure(input2_closure))
        return rewire_to_input("in2");
    if (!is_empty_closure(input1_closure) && is_empty_closure(input2_closure))
        return rewire_to_input("in1");
    return keep_node();
}

NodeClassification classify_layer(const mx::ShaderNode& node, const OutputStaticValueMap& current)
{
    const ClosureStaticValue top_closure = closure_of(node.getInput("top"), current);
    const ClosureStaticValue base_closure = closure_of(node.getInput("base"), current);
    if (is_empty_closure(top_closure) && is_empty_closure(base_closure))
        return remove_as_empty();
    if (is_empty_closure(top_closure) && !is_empty_closure(base_closure))
        return rewire_to_input("base");
    if (!is_empty_closure(top_closure) && is_empty_closure(base_closure))
        return rewire_to_input("top");
    return keep_node();
}

NodeClassification classify_weighted_layer(const mx::ShaderNode& node, const OutputStaticValueMap& current)
{
    const ClosureStaticValue top_closure = closure_of(node.getInput("top"), current);
    const ClosureStaticValue base_closure = closure_of(node.getInput("base"), current);
    const mx::ShaderInput* weight_input = node.getInput("top_weight");
    const ScalarStaticValue top_weight
        = weight_input ? scalar_of(*weight_input, current) : ScalarStaticValue::unknown();
    const bool top_is_empty = is_empty_closure(top_closure);
    const bool base_is_empty = is_empty_closure(base_closure);
    if (top_is_empty && base_is_empty)
        return remove_as_empty();
    if (top_is_empty && !base_is_empty)
        return rewire_to_input("base");
    if (is_zero_static_value(top_weight))
        return base_is_empty ? remove_as_empty() : rewire_to_input("base");
    if (!top_is_empty && base_is_empty && is_one_static_value(top_weight))
        return rewire_to_input("top");
    return keep_node();
}

NodeClassification classify_multiply_closure(const mx::ShaderNode& node, const OutputStaticValueMap& current)
{
    // in1 = closure, in2 = float/color scalar
    const ClosureStaticValue closure = closure_of(node.getInput("in1"), current);
    const ScalarStaticValue factor = scalar_of(node.getInput("in2"), current);
    if (is_zero_static_value(factor) || is_empty_closure(closure))
        return remove_as_empty();
    if (is_one_static_value(factor))
        return rewire_to_input("in1");
    return keep_node();
}

void classify_node_into(const mx::ShaderNode& node, OutputStaticValueMap& current, ClosurePruningAnalysisResult& result)
{
    const std::string_view impl = shader_node_implementation_name(node);

    // Compound node: propagate inner-graph output-socket values to this
    // node's outputs. Classified inner graphs were processed first (we
    // walk graphs in reverse reachable order), so inner output sockets
    // already have entries in `current`.
    if (const mx::ShaderGraph* inner = node.getImplementation().getGraph()) {
        for (size_t i = 0; i < node.numOutputs(); ++i) {
            const mx::ShaderOutput* outer_out = node.getOutput(i);
            const mx::ShaderInput* inner_sink = (i < inner->numOutputSockets()) ? inner->getOutputSocket(i) : nullptr;
            const mx::ShaderOutput* inner_src = inner_sink ? inner_sink->getConnection() : nullptr;
            OutputStaticValue v;
            if (inner_src) {
                v = output_static_value_or_unknown(inner_src, current);
            } else {
                // Inner output socket has no driver; treat as KnownEmpty
                // closure / Unknown scalar.
                v.closure = ClosureStaticValue::KnownEmpty;
                v.scalar = ScalarStaticValue::unknown();
            }
            current[outer_out] = v;
        }
        return;
    }

    // Closure ops.
    bool has_asymmetric_mix_fold_plan = false;
    AsymmetricMixFoldPlan asymmetric_mix_fold_plan;
    NodeClassification classification;
    bool has_closure_rule = false;

    if (is_connected_layer_vdf_boundary(node)) {
        // A connected VDF base marks a volume boundary. Keep the layer_vdf
        // node itself live, then allow ordinary BSDF pruning above it.
        classification = keep_node();
        has_closure_rule = true;
    } else if (is_layer_vdf_node(node)) {
        // Missing/unconnected VDF base is not a live boundary.
        classification = classify_layer(node, current);
        has_closure_rule = true;
    } else if (matches_surface_closure_op(impl, "mix")) {
        const MixClosureClassification mix_result = classify_mix_closure(node, current);
        classification = mix_result.classification;
        has_asymmetric_mix_fold_plan = mix_result.has_asymmetric_mix_fold_plan;
        asymmetric_mix_fold_plan = mix_result.asymmetric_mix_fold_plan;
        has_closure_rule = true;
    } else if (matches_surface_closure_op(impl, "add")) {
        classification = classify_add_closure(node, current);
        has_closure_rule = true;
    } else if (matches_surface_closure_op(impl, "layer")) {
        classification = classify_layer(node, current);
        has_closure_rule = true;
    } else if (is_weighted_layer_node(node)) {
        classification = classify_weighted_layer(node, current);
        has_closure_rule = true;
    } else if (matches_surface_closure_op(impl, "multiply")) {
        classification = classify_multiply_closure(node, current);
        has_closure_rule = true;
    }

    // Generic leaf-closure rule: a closure-typed node with no closure-typed
    // inputs and a scalar `weight` input that proves KnownZero is KnownEmpty.
    // Covers sheen_bsdf, conductor_bsdf, oren_nayar_diffuse_bsdf, etc.
    if (!has_closure_rule && node.numOutputs() == 1) {
        const mx::TypeDesc out_type = node.getOutput(0)->getType();
        const bool is_closure_out = is_surface_closure_type(out_type);
        const mx::ShaderInput* w = node.getInput("weight");
        if (is_closure_out && w != nullptr && is_supported_weight_value_type(w->getType())) {
            bool any_closure_input = false;
            for (size_t i = 0; i < node.numInputs(); ++i) {
                const mx::TypeDesc t = node.getInput(i)->getType();
                if (is_any_closure_type(t)) {
                    any_closure_input = true;
                    break;
                }
            }
            if (!any_closure_input && is_zero_static_value(scalar_of(*w, current))) {
                classification = remove_as_empty();
                has_closure_rule = true;
            }
        }
    }

    // Scalar / color ops that propagate constants.
    ScalarStaticValue scalar_out = ScalarStaticValue::unknown();
    bool has_scalar_rule = false;

    if (!has_closure_rule) {
        if (is_constant_node(impl)) {
            // Output value lifts directly from the `value` input/default.
            if (const mx::ShaderInput* v = node.getInput("value")) {
                scalar_out = scalar_of(*v, current);
                has_scalar_rule = true;
            }
        } else if (is_scalar_or_color_multiply_node(impl)) {
            const ScalarStaticValue input1_value = scalar_of(node.getInput("in1"), current);
            const ScalarStaticValue input2_value = scalar_of(node.getInput("in2"), current);
            if (is_zero_static_value(input1_value) || is_zero_static_value(input2_value))
                scalar_out = ScalarStaticValue::zero();
            else if (is_one_static_value(input1_value))
                scalar_out = input2_value;
            else if (is_one_static_value(input2_value))
                scalar_out = input1_value;
            else
                scalar_out = ScalarStaticValue::unknown();
            has_scalar_rule = true;
        } else if (is_invert_float_node(impl)) {
            const ScalarStaticValue input_value = scalar_of(node.getInput("in"), current);
            if (is_zero_static_value(input_value))
                scalar_out = ScalarStaticValue::one();
            else if (is_one_static_value(input_value))
                scalar_out = ScalarStaticValue::zero();
            else
                scalar_out = ScalarStaticValue::unknown();
            has_scalar_rule = true;
        } else if (is_dot_node(impl)) {
            scalar_out = scalar_of(node.getInput("in"), current);
            has_scalar_rule = true;
        } else if (is_add_float_node(impl)) {
            const ScalarStaticValue input1_value = scalar_of(node.getInput("in1"), current);
            const ScalarStaticValue input2_value = scalar_of(node.getInput("in2"), current);
            if (is_zero_static_value(input1_value))
                scalar_out = input2_value;
            else if (is_zero_static_value(input2_value))
                scalar_out = input1_value;
            else
                scalar_out = ScalarStaticValue::unknown();
            has_scalar_rule = true;
        }
    }

    // Populate output_static_values for every output of this node.
    for (size_t i = 0; i < node.numOutputs(); ++i) {
        const mx::ShaderOutput* out = node.getOutput(i);
        OutputStaticValue v;
        if (has_closure_rule) {
            // Closure node: closure-side from classification, scalar Unknown.
            if (classification.kind == NodeClassification::Kind::RemoveAsEmpty)
                v.closure = ClosureStaticValue::KnownEmpty;
            else if (classification.kind == NodeClassification::Kind::RewireToInput)
                v.closure = closure_of(node.getInput(classification.rewire_input), current);
            else
                v.closure = ClosureStaticValue::KnownNonEmpty;
            v.scalar = ScalarStaticValue::unknown();
        } else if (has_scalar_rule) {
            v.closure = ClosureStaticValue::Unknown;
            v.scalar = scalar_out;
        } else {
            v.closure = ClosureStaticValue::Unknown;
            v.scalar = ScalarStaticValue::unknown();
        }
        current[out] = v;
    }

    if (has_closure_rule) {
        result.pruning.classification[&node] = classification;
        if (has_asymmetric_mix_fold_plan && classification.kind == NodeClassification::Kind::Keep) {
            result.pruning.asymmetric_mix_fold_plans[&node] = asymmetric_mix_fold_plan;
        }
    }
}

} // namespace

ClosurePruningAnalysisResult analyze_closure_pruning(
    const mx::ShaderGraph& top,
    const CallSiteIndex& index,
    const OutputStaticValueMap& prev_output_static_values
)
{
    ClosurePruningAnalysisResult result;
    OutputStaticValueMap& current = result.static_values.output_static_values;

    // Seed environments for every reachable graph.
    for (const mx::ShaderGraph* graph : index.reachable_graphs()) {
        if (graph == &top) {
            seed_top_graph_env(top, current);
        } else {
            const std::vector<CallSite>& sites = index.call_sites_of(graph);
            seed_inner_graph_env(*graph, sites, prev_output_static_values, current);
        }
    }

    // Classify nodes within each graph in topological (getNodes) order.
    // Walk graphs leaves-first so a compound node in an outer graph can read
    // its inner graph's already-classified output sockets in the same call.
    const auto& graphs = index.reachable_graphs();
    for (auto it = graphs.rbegin(); it != graphs.rend(); ++it) {
        for (const mx::ShaderNode* node : (*it)->getNodes()) {
            classify_node_into(*node, current, result);
        }
    }

    return result;
}

StaticValueAnalysisResult analyze_static_values(
    const mx::ShaderGraph& top,
    const CallSiteIndex& index,
    const OutputStaticValueMap& prev_output_static_values
)
{
    return analyze_closure_pruning(top, index, prev_output_static_values).static_values;
}

StaticValueAnalysisResult analyze_static_values_to_fixpoint(const mx::ShaderGraph& top)
{
    OutputStaticValueMap prev_values;
    StaticValueAnalysisResult result;

    const CallSiteIndex initial_index = build_call_site_index(top);
    const size_t iteration_cap = 2 * initial_index.reachable_graphs().size() + initial_index.reachable_node_count();

    size_t iterations = 0;
    while (true) {
        if (iterations >= iteration_cap) {
            FALCOR_THROW(
                "static value analysis exceeded iteration cap ({}) on graph '{}'",
                iteration_cap,
                top.getName()
            );
        }

        const CallSiteIndex index = build_call_site_index(top);
        result = analyze_static_values(top, index, prev_values);
        ++iterations;

        if (result.output_static_values == prev_values)
            break;

        prev_values = result.output_static_values;
    }

    return result;
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
