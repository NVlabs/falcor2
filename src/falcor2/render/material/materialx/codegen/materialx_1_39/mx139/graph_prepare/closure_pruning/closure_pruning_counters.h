// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace graph_prepare {

struct ClosurePruningCounters {
    bool requested_enabled = true;
    bool effective_enabled = false;

    // Successful output replacements performed by the pruning pass. This
    // counts the rewrite action, not the number of consumer sockets that
    // MaterialX redirects internally.
    uint32_t output_rewrites = 0;

    // Output rewrites caused specifically by asymmetric mix materialization:
    // mix(empty, live, w) -> multiply(live, 1 - w), or
    // mix(live, empty, w) -> multiply(live, w). This is a subset of
    // output_rewrites, not a separate count to add on top.
    uint32_t asymmetric_mix_rewrites = 0;

    uint32_t nodes_pruned = 0;
    uint32_t iterations = 0;
};

} // namespace graph_prepare
} // namespace mx139
} // namespace materialx
} // namespace falcor
