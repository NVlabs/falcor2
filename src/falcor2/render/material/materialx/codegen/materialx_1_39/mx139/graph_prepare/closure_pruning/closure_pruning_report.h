// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "closure_pruning_counters.h"

#include "falcor2/core/properties.h"

#include <ostream>
#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace graph_prepare {

inline constexpr const char* k_closure_pruning_requested_mode = "closure_pruning_requested_mode";
inline constexpr const char* k_closure_pruning_effective_mode = "closure_pruning_effective_mode";
inline constexpr const char* k_closure_pruning_effective_enabled = "closure_pruning_effective_enabled";
inline constexpr const char* k_closure_pruning_output_rewrites = "closure_pruning_output_rewrites";
inline constexpr const char* k_closure_pruning_asymmetric_mix_rewrites = "closure_pruning_asymmetric_mix_rewrites";
inline constexpr const char* k_closure_pruning_nodes_pruned = "closure_pruning_nodes_pruned";
inline constexpr const char* k_closure_pruning_iterations = "closure_pruning_iterations";

inline constexpr const char* closure_pruning_enabled_to_string(bool enabled)
{
    return enabled ? "enabled" : "disabled";
}

inline void write_closure_pruning_report_fields(Properties& props, const ClosurePruningCounters& counters)
{
    props.set(
        k_closure_pruning_requested_mode,
        std::string(closure_pruning_enabled_to_string(counters.requested_enabled))
    );
    props.set(
        k_closure_pruning_effective_mode,
        std::string(closure_pruning_enabled_to_string(counters.effective_enabled))
    );
    props.set(k_closure_pruning_effective_enabled, counters.effective_enabled);
    props.set(k_closure_pruning_output_rewrites, static_cast<int>(counters.output_rewrites));
    props.set(k_closure_pruning_asymmetric_mix_rewrites, static_cast<int>(counters.asymmetric_mix_rewrites));
    props.set(k_closure_pruning_nodes_pruned, static_cast<int>(counters.nodes_pruned));
    props.set(k_closure_pruning_iterations, static_cast<int>(counters.iterations));
}

inline void write_closure_pruning_report_tsv(std::ostream& out, const Properties& props)
{
    out << "\t" << props.get<std::string>(k_closure_pruning_requested_mode);
    out << "\t" << props.get<std::string>(k_closure_pruning_effective_mode);
    out << "\t" << (props.get<bool>(k_closure_pruning_effective_enabled) ? "1" : "0");
    out << "\t" << props.get<int>(k_closure_pruning_output_rewrites);
    out << "\t" << props.get<int>(k_closure_pruning_asymmetric_mix_rewrites);
    out << "\t" << props.get<int>(k_closure_pruning_nodes_pruned);
    out << "\t" << props.get<int>(k_closure_pruning_iterations);
}

} // namespace graph_prepare
} // namespace mx139
} // namespace materialx
} // namespace falcor
