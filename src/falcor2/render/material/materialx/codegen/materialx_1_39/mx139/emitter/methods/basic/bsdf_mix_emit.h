// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "composition_aliases.h"

#include "../../../layering_analysis.h"
#include "../../../profile_policy.h"

#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

// Emits the generated bsdf_mix Mx139 root BSDF type. This consumes the
// layering analysis and profile naming policy only; it does not mutate graph or
// layering state.
void emit_bsdf_mix_root_bsdf_type(
    const MaterialX::ShaderGenerator& shadergen,
    const LayeringDesc& layering,
    const ProfileDesc& profile,
    const std::string& bsdf_mix_root_type,
    MaterialX::ShaderStage& stage
);

// Emits the local bsdf_mix root value used by the basic material
// wrapper and returns the Slang expression/name to store in MaterialInstance.
std::string emit_bsdf_mix_composition_value(
    const MaterialX::ShaderGenerator& shadergen,
    const LayeringDesc& layering,
    const CompositionAliases& aliases,
    const std::string& stack_name,
    const std::string& wi_expr,
    MaterialX::ShaderStage& stage,
    bool use_local_aliases
);

// Same as `emit_bsdf_mix_composition_value` but appends the multi-line
// `let root_bsdf = AliasName(...)` call to `out` (each line prefixed by
// `line_prefix`) instead of writing through the MaterialX shader stage. Used
// by the wrapper template renderer in Step 7.
std::string build_bsdf_mix_composition_value_text(
    const LayeringDesc& layering,
    const CompositionAliases& aliases,
    const std::string& stack_name,
    const std::string& wi_expr,
    bool use_local_aliases,
    const std::string& line_prefix,
    std::string& out
);

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
