// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "classify.h"
#include "closure_pruning_counters.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGraph.h>

namespace falcor {
namespace materialx {
namespace mx139 {

// Apply one round of closure simplification driven by `cr`. Returns true when
// the graph mutated, so the driver should re-run analysis before deciding
// whether the repeat loop is stable. Counter fields on `out` are incremented
// in place; the driver owns iteration accounting.
bool apply_closure_simplification(
    const ClosurePruningAnalysisResult& cr,
    const CallSiteIndex& index,
    MaterialX::GenContext& context,
    graph_prepare::ClosurePruningCounters& out
);

} // namespace mx139
} // namespace materialx
} // namespace falcor
