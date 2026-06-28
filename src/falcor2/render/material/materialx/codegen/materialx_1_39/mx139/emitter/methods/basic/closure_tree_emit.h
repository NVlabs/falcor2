// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "composition_aliases.h"

#include "../../../layering_analysis.h"
#include "../../../profile_policy.h"

#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <cstddef>
#include <map>
#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

// Emits the type-alias namespace consumed by the closure_tree composition path.
// This is source emission only; it does not modify the MaterialX graph or the
// layering analysis.
void emit_closure_tree_composition_aliases(
    const MaterialX::ShaderGenerator& shadergen,
    const CompositionAliases& aliases,
    const ProfileDesc& profile,
    MaterialX::ShaderStage& stage
);

// Emits a closure_tree recursive BSDF composition value and returns the generated
// expression/name. `emitted_values` caches subtrees so shared references keep
// stable local names within one wrapper body.
std::string emit_closure_tree_composition_value(
    const MaterialX::ShaderGenerator& shadergen,
    const LayeringDesc& layering,
    const CompositionAliases& aliases,
    ClosureRef ref,
    const std::string& stack_name,
    std::map<ClosureRef, std::string>& emitted_values,
    MaterialX::ShaderStage& stage,
    bool use_local_aliases
);

// Same as `emit_closure_tree_composition_value` but appends the recursive
// `let ... = ...;` cascade to `out` (each line prefixed by `line_prefix` and
// terminated by a newline) instead of writing through the MaterialX shader
// stage. Returns the same expression/name the emit variant returns. Used by
// the wrapper template renderer in Step 7 to splice the cascade into a
// $-token slot.
std::string build_closure_tree_composition_value_text(
    const LayeringDesc& layering,
    const CompositionAliases& aliases,
    ClosureRef ref,
    const std::string& stack_name,
    bool use_local_aliases,
    const std::string& line_prefix,
    std::map<ClosureRef, std::string>& emitted_values,
    std::string& out
);

// Inputs for the closure_tree/bsdf_mix material wrapper. The emitter may use
// bsdf_mix helpers when that option is selected, but it owns the material
// wrapper shape used by closure_tree/bsdf_mix.
struct ClosureTreeMaterialStructsDesc {
    std::string material_name;
    std::string material_instance_name;
    std::string material_data_name;
    std::string graph_name;
    const LayeringDesc& layering;
    const CodeGenDesc& codegen_desc;
    const ProfileDesc& profile;
    bool graph_has_emission = false;
    std::size_t materialx_payload_size = 0;
};

void emit_closure_tree_material_structs(
    const MaterialX::ShaderGenerator& shadergen,
    const ClosureTreeMaterialStructsDesc& params,
    MaterialX::ShaderStage& stage
);

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
