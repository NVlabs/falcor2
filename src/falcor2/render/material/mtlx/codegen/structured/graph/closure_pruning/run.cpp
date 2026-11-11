// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "run.h"

#include "classify.h"
#include "closure_simplification.h"
#include "../static_input_query/call_site_index.h"

#include "falcor2/core/error.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGraph.h>

namespace falcor {
namespace mtlx::structured {

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

void disconnect_known_empty_shader_inputs(
    MaterialX::ShaderGraph& graph,
    const ClosurePruningAnalysisResult& analysis,
    ClosurePruningCounters& counters
)
{
    bool mutated = false;
    for (MaterialX::ShaderNode* node : graph.getNodes()) {
        if (!node)
            continue;
        if (!produces_surface_shader(*node))
            continue;
        for (size_t i = 0; i < node->numInputs(); ++i) {
            MaterialX::ShaderInput* input = node->getInput(i);
            if (!input || !is_surface_closure_input(*input))
                continue;
            const MaterialX::ShaderOutput* driver = input->getConnection();
            if (!driver)
                continue;
            const auto value_it = analysis.static_values.output_static_values.find(driver);
            if (value_it == analysis.static_values.output_static_values.end()
                || value_it->second.closure != ClosureStaticValue::KnownEmpty)
                continue;
            input->breakConnection();
            mutated = true;
        }
    }
    if (!mutated)
        return;
    const size_t before = graph.getNodes().size();
    graph.removeUnusedNodes();
    graph.topologicalSort();
    const size_t after = graph.getNodes().size();
    counters.nodes_pruned += before > after ? static_cast<uint32_t>(before - after) : 0u;
}
} // namespace

void run_closure_pruning(
    MaterialX::ShaderGraph& graph,
    MaterialX::GenContext& context,
    const CodeGenDesc& desc,
    const RuntimeInputPredicate& is_runtime_input,
    ClosurePruningCounters& result
)
{
    result.requested_enabled = is_set(desc.optimize_graph_flags, OptimizeGraphFlags::closure_pruning);
    result.effective_enabled = false;
    result.output_rewrites = 0;
    result.asymmetric_mix_rewrites = 0;
    result.nodes_pruned = 0;
    result.iterations = 0;

    if (!result.requested_enabled) {
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
        analysis = analyze_closure_pruning(graph, index, prev_values, is_runtime_input);
        const bool mutated = apply_closure_simplification(analysis, index, context, result);
        ++result.iterations;
        const bool pruning_classification_stable = (analysis.pruning.classification == prev_classification);
        const bool static_values_stable = (analysis.static_values.output_static_values == prev_values);
        prev_values = analysis.static_values.output_static_values;
        prev_classification = analysis.pruning.classification;
        if (!mutated && pruning_classification_stable && static_values_stable)
            break;
    }

    // TODO(@tdavidovic): Investigate returning the final static-value analysis
    // so post-pruning consumers such as opacity and Fresnel can reuse it.

    // Make an empty surface result visible to the explicit
    // EmptySurfaceHandler by disconnecting only inputs proven KnownEmpty.
    // EDF and VDF inputs not proven empty remain intact.
    const CallSiteIndex final_index = build_call_site_index(graph);
    for (const MaterialX::ShaderGraph* reachable : final_index.reachable_graphs()) {
        disconnect_known_empty_shader_inputs(const_cast<MaterialX::ShaderGraph&>(*reachable), analysis, result);
    }
}

} // namespace mtlx::structured
} // namespace falcor
