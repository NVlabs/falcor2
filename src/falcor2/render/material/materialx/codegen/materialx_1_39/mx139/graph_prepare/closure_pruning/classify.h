// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../static_input_query/call_site_index.h"
#include "../static_input_query/static_value_analysis.h"

#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderNode.h>

#include <string>
#include <unordered_map>

namespace falcor {
namespace materialx {
namespace mx139 {

// Classification of a single ShaderNode.
struct NodeClassification {
    enum class Kind : uint8_t { RemoveAsEmpty, Keep, RewireToInput };
    Kind kind = Kind::Keep;
    // Valid when kind == RewireToInput: the name of the input socket on this
    // node whose connected upstream output should be used in place of this
    // node's output.
    std::string rewire_input;
};

inline bool operator==(const NodeClassification& a, const NodeClassification& b)
{
    return a.kind == b.kind && a.rewire_input == b.rewire_input;
}
inline bool operator!=(const NodeClassification& a, const NodeClassification& b)
{
    return !(a == b);
}

// Side-table plan for asymmetric `mix` rewrites (one branch is known empty,
// the other is kept). The pruning consumer reads this to decide which side to
// keep and which weight socket to fold in.
struct AsymmetricMixFoldPlan {
    enum class KeptSide : uint8_t { Fg, Bg };
    KeptSide kept_side = KeptSide::Fg;
};

using NodeClassificationMap = std::unordered_map<const MaterialX::ShaderNode*, NodeClassification>;
using AsymmetricMixFoldPlanMap = std::unordered_map<const MaterialX::ShaderNode*, AsymmetricMixFoldPlan>;

// Pruning-specific side data derived by the mutation-free analysis pass.
// Graph mutation, helper materialization, removeUnusedNodes(), and counter
// updates stay in the pruning consumer.
struct ClosurePruningAnalysis {
    NodeClassificationMap classification;
    AsymmetricMixFoldPlanMap asymmetric_mix_fold_plans;
};

struct ClosurePruningAnalysisResult {
    StaticValueAnalysisResult static_values;
    ClosurePruningAnalysis pruning;
};

// Mutation-free solver. Walks every reachable graph in `index` (top graph
// has index 0), derives a per-graph input environment top-down from
// `prev_output_static_values`, and produces reusable static values plus
// pruning-specific classifications/rewrite plans. `prev_output_static_values`
// is the result of the previous iteration's classification; pass an empty map
// on iteration zero.
//
// The solver MUST NOT mutate any ShaderGraph and takes no user_data.
ClosurePruningAnalysisResult analyze_closure_pruning(
    const MaterialX::ShaderGraph& top,
    const CallSiteIndex& index,
    const OutputStaticValueMap& prev_output_static_values
);

} // namespace mx139
} // namespace materialx
} // namespace falcor
