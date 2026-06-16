// SPDX-License-Identifier: Apache-2.0

#include "run.h"

#include "classify.h"
#include "closure_simplification.h"
#include "../static_input_query/call_site_index.h"

#include "../../codegen_user_data.h"
#include "../../policy.h"

#include "falcor2/core/error.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGraph.h>

namespace falcor {
namespace materialx {
namespace mx139 {

namespace {
bool produces_surface_shader(const MaterialX::ShaderNode& node)
{
    for (size_t i = 0; i < node.numOutputs(); ++i) {
        const MaterialX::TypeDesc type = node.getOutput(i)->getType();
        if (type == MaterialX::Type::SURFACESHADER || type == MaterialX::Type::DISPLACEMENTSHADER
            || type == MaterialX::Type::LIGHTSHADER) {
            return true;
        }
    }
    return false;
}

bool is_surface_closure_input(const MaterialX::ShaderInput& input)
{
    const MaterialX::TypeDesc type = input.getType();
    return type == MaterialX::Type::BSDF || type == MaterialX::Type::EDF;
}

bool has_only_empty_connected_surface_closure_inputs(
    const MaterialX::ShaderNode& node,
    const OutputStaticValueMap& output_static_values
)
{
    bool has_connected_surface_closure = false;
    for (size_t i = 0; i < node.numInputs(); ++i) {
        const MaterialX::ShaderInput* input = node.getInput(i);
        if (!input || !is_surface_closure_input(*input))
            continue;

        const MaterialX::ShaderOutput* driver = input->getConnection();
        if (!driver)
            continue;

        has_connected_surface_closure = true;
        const auto value_it = output_static_values.find(driver);
        if (value_it == output_static_values.end() || value_it->second.closure != ClosureStaticValue::KnownEmpty)
            return false;
    }
    return has_connected_surface_closure;
}

void throw_if_shader_constructor_has_only_empty_surface_closures(
    const MaterialX::ShaderGraph& graph,
    const ClosurePruningAnalysisResult& analysis
)
{
    for (const MaterialX::ShaderNode* node : graph.getNodes()) {
        if (!node)
            continue;
        if (!produces_surface_shader(*node))
            continue;
        if (!has_only_empty_connected_surface_closure_inputs(*node, analysis.static_values.output_static_values))
            continue;

        FALCOR_THROW(
            "closure_pruning: shader constructor '{}' has only KnownEmpty closure inputs in graph '{}'",
            node->getName(),
            graph.getName()
        );
    }
}
} // namespace

void run_closure_pruning(MaterialX::ShaderGraph& graph, MaterialX::GenContext& context, CodegenUserData& user_data)
{
    graph_prepare::ClosurePruningCounters& result = user_data.analysis.closure_pruning;
    result.requested_enabled = is_set(user_data.inputs.desc->optimize_graph_flags, OptimizeGraphFlags::closure_pruning);
    result.effective_enabled = false;
    result.output_rewrites = 0;
    result.asymmetric_mix_rewrites = 0;
    result.nodes_pruned = 0;
    result.iterations = 0;

    const OptimizeGraphFlags effective_flags = effective_optimize_graph_flags(*user_data.inputs.desc);
    if (!is_set(effective_flags, OptimizeGraphFlags::closure_pruning)) {
        return;
    }

    result.effective_enabled = true;

    // Termination: stop when classification and static values are stable AND
    // no mutation occurred this iteration. Static values can lag through
    // shared graph inputs, so a no-mutation pass may still expose a pruning
    // proof on the next iteration. The cap is a safety net; overruns signal a
    // bug in the classify/prune pair and must crash, not wedge.
    OutputStaticValueMap prev_values;
    NodeClassificationMap prev_classification;
    const CallSiteIndex initial_index = build_call_site_index(graph);
    const size_t iteration_cap = 2 * initial_index.reachable_graphs().size() + initial_index.reachable_node_count();

    ClosurePruningAnalysisResult analysis;
    while (true) {
        if (result.iterations >= iteration_cap) {
            FALCOR_THROW(
                "closure_pruning driver exceeded iteration cap ({}) on graph '{}'",
                iteration_cap,
                graph.getName()
            );
        }
        const CallSiteIndex index = build_call_site_index(graph);
        analysis = analyze_closure_pruning(graph, index, prev_values);
        const bool mutated = apply_closure_simplification(analysis, index, context, result);
        ++result.iterations;
        const bool pruning_classification_stable = (analysis.pruning.classification == prev_classification);
        const bool static_values_stable = (analysis.static_values.output_static_values == prev_values);
        prev_values = analysis.static_values.output_static_values;
        prev_classification = analysis.pruning.classification;
        if (!mutated && pruning_classification_stable && static_values_stable)
            break;
    }

    // Top-shader collapse: if a non-volume shader-constructor node (output type
    // SURFACESHADER / DISPLACEMENTSHADER / LIGHTSHADER) has
    // any connected closure-typed input and ALL such inputs classify
    // KnownEmpty,
    // fail fast -- downstream codegen and runtime normalize selection
    // weights and would divide by zero. VDF-typed inputs and VOLUMESHADER
    // producers are not BSDF/EDF surface channels, so this diagnostic leaves
    // them to the volume path.
    throw_if_shader_constructor_has_only_empty_surface_closures(graph, analysis);
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
