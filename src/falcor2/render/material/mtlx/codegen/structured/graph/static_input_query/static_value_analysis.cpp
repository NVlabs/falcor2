// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "static_value_analysis.h"

#include "../closure_pruning/classify.h"

#include "falcor2/core/error.h"

namespace falcor {
namespace mtlx::structured {

namespace mx = MaterialX;

StaticValueAnalysisResult
analyze_graph_static_values(const mx::ShaderGraph& top, const RuntimeInputPredicate& is_runtime_input)
{
    OutputStaticValueMap previous_values;
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
        result = analyze_closure_pruning(top, index, previous_values, is_runtime_input).static_values;
        ++iterations;

        if (result.output_static_values == previous_values)
            break;

        previous_values = result.output_static_values;
    }

    return result;
}

} // namespace mtlx::structured
} // namespace falcor
