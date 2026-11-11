// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../../codegen_types.h"
#include "closure_pruning_counters.h"
#include "../static_input_query/static_value_analysis.h"

#include "falcor2/core/macros.h"

#include <cstdint>

namespace MaterialX_v1_39_5 {
class ShaderGraph;
class GenContext;
} // namespace MaterialX_v1_39_5
namespace MaterialX = MaterialX_v1_39_5;

namespace falcor {
namespace mtlx::structured {

/// Iteratively analyze and rewrite statically empty closure subgraphs. The
/// pass mutates graph topology before the structured pipeline freezes it and
/// reports every effective rewrite through result.
FALCOR_API void run_closure_pruning(
    MaterialX::ShaderGraph& graph,
    MaterialX::GenContext& context,
    const CodeGenDesc& desc,
    const RuntimeInputPredicate& is_runtime_input,
    ClosurePruningCounters& result
);

} // namespace mtlx::structured
} // namespace falcor
