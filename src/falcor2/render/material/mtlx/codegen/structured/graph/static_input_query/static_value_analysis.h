// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "static_values.h"

#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderNode.h>

#include <functional>
#include <string>
#include <unordered_map>

namespace falcor {
namespace mtlx::structured {

// Per-output static value.
struct OutputStaticValue {
    ClosureStaticValue closure = ClosureStaticValue::Unknown;
    ScalarStaticValue scalar = ScalarStaticValue::unknown();

    bool operator==(const OutputStaticValue&) const = default;
};

inline OutputStaticValue merge_output_static_values(const OutputStaticValue& a, const OutputStaticValue& b)
{
    return OutputStaticValue{merge_closure_values(a.closure, b.closure), merge_scalar_values(a.scalar, b.scalar)};
}

using OutputStaticValueMap = std::unordered_map<const MaterialX::ShaderOutput*, OutputStaticValue>;
using RuntimeInputPredicate = std::function<bool(const MaterialX::ShaderOutput&)>;

// Reusable static-value facts from a mutation-free graph analysis pass.
// Future consumers such as Fresnel policy selection should depend on this
// type instead of the pruning rewrite data.
struct StaticValueAnalysisResult {
    OutputStaticValueMap output_static_values;
};

// Analyze static values across the graph and all reachable nodegraphs. The
// implementation iterates until values crossing graph interfaces converge; it
// does not apply pruning rewrites or update pruning counters.
StaticValueAnalysisResult
analyze_graph_static_values(const MaterialX::ShaderGraph& top, const RuntimeInputPredicate& is_runtime_input);

} // namespace mtlx::structured
} // namespace falcor
