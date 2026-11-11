// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "closure_pruning_counters.h"

#include "falcor2/core/properties.h"

#include <ostream>
#include <string>

namespace falcor {
namespace mtlx::structured {

inline constexpr const char* CLOSURE_PRUNING_REQUESTED_MODE = "closure_pruning_requested_mode";
inline constexpr const char* CLOSURE_PRUNING_EFFECTIVE_MODE = "closure_pruning_effective_mode";
inline constexpr const char* CLOSURE_PRUNING_EFFECTIVE_ENABLED = "closure_pruning_effective_enabled";
inline constexpr const char* CLOSURE_PRUNING_OUTPUT_REWRITES = "closure_pruning_output_rewrites";
inline constexpr const char* CLOSURE_PRUNING_ASYMMETRIC_MIX_REWRITES = "closure_pruning_asymmetric_mix_rewrites";
inline constexpr const char* CLOSURE_PRUNING_NODES_PRUNED = "closure_pruning_nodes_pruned";
inline constexpr const char* CLOSURE_PRUNING_ITERATIONS = "closure_pruning_iterations";

inline constexpr const char* closure_pruning_enabled_to_string(bool enabled)
{
    return enabled ? "enabled" : "disabled";
}

inline void write_closure_pruning_report_fields(Properties& props, const ClosurePruningCounters& counters)
{
    props.set(
        CLOSURE_PRUNING_REQUESTED_MODE,
        std::string(closure_pruning_enabled_to_string(counters.requested_enabled))
    );
    props.set(
        CLOSURE_PRUNING_EFFECTIVE_MODE,
        std::string(closure_pruning_enabled_to_string(counters.effective_enabled))
    );
    props.set(CLOSURE_PRUNING_EFFECTIVE_ENABLED, counters.effective_enabled);
    props.set(CLOSURE_PRUNING_OUTPUT_REWRITES, static_cast<int>(counters.output_rewrites));
    props.set(CLOSURE_PRUNING_ASYMMETRIC_MIX_REWRITES, static_cast<int>(counters.asymmetric_mix_rewrites));
    props.set(CLOSURE_PRUNING_NODES_PRUNED, static_cast<int>(counters.nodes_pruned));
    props.set(CLOSURE_PRUNING_ITERATIONS, static_cast<int>(counters.iterations));
}

inline void write_closure_pruning_report_tsv(std::ostream& out, const Properties& props)
{
    out << "\t" << props.get<std::string>(CLOSURE_PRUNING_REQUESTED_MODE);
    out << "\t" << props.get<std::string>(CLOSURE_PRUNING_EFFECTIVE_MODE);
    out << "\t" << (props.get<bool>(CLOSURE_PRUNING_EFFECTIVE_ENABLED) ? "1" : "0");
    out << "\t" << props.get<int>(CLOSURE_PRUNING_OUTPUT_REWRITES);
    out << "\t" << props.get<int>(CLOSURE_PRUNING_ASYMMETRIC_MIX_REWRITES);
    out << "\t" << props.get<int>(CLOSURE_PRUNING_NODES_PRUNED);
    out << "\t" << props.get<int>(CLOSURE_PRUNING_ITERATIONS);
}

} // namespace mtlx::structured
} // namespace falcor
