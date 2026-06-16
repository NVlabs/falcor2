// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "call_site_index.h"
#include "static_values.h"

#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderNode.h>

#include <unordered_map>

namespace falcor {
namespace materialx {
namespace mx139 {

// Per-output static value.
struct OutputStaticValue {
    ClosureStaticValue closure = ClosureStaticValue::Unknown;
    ScalarStaticValue scalar = ScalarStaticValue::unknown();

    bool operator==(const OutputStaticValue& o) const { return closure == o.closure && scalar == o.scalar; }
    bool operator!=(const OutputStaticValue& o) const { return !(*this == o); }
};

inline OutputStaticValue merge_output_static_values(const OutputStaticValue& a, const OutputStaticValue& b)
{
    return OutputStaticValue{merge_closure_values(a.closure, b.closure), merge_scalar_values(a.scalar, b.scalar)};
}

using OutputStaticValueMap = std::unordered_map<const MaterialX::ShaderOutput*, OutputStaticValue>;

// Reusable static-value facts from a mutation-free graph analysis pass.
// Future consumers such as Fresnel policy selection should depend on this
// type instead of the pruning rewrite data.
struct StaticValueAnalysisResult {
    OutputStaticValueMap output_static_values;
};

// Mutation-free static-value analysis entry point for non-pruning consumers.
// It may compute pruning classifications internally, but callers only depend on
// the static values returned here.
StaticValueAnalysisResult analyze_static_values(
    const MaterialX::ShaderGraph& top,
    const CallSiteIndex& index,
    const OutputStaticValueMap& prev_output_static_values
);

// Mutation-free fixed-point analysis for non-pruning consumers.
// Repeats static-value analysis until values stop changing across graph
// interface boundaries. This does not apply pruning rewrites or update pruning
// counters.
StaticValueAnalysisResult analyze_static_values_to_fixpoint(const MaterialX::ShaderGraph& top);

} // namespace mx139
} // namespace materialx
} // namespace falcor
