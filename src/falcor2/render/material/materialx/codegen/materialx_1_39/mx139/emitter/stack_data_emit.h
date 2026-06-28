// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../layering_analysis.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <cstddef>
#include <functional>
#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

// Callback that emits the `public ...` member declarations for a uniform
// block (uses the generator's syntax / geomprop policy). Injected so this
// emitter stays free of the GenslangPt shader generator's private state.
using EmitShaderInputsFn = std::function<
    void(const MaterialX::VariableBlock&, MaterialX::ShaderStage&, bool public_members, bool as_property)>;

// Callback that returns the packed byte size of a uniform block, matching
// the rules the generator uses elsewhere for `data_struct_size`.
using CalculateByteSizeFn = std::function<std::size_t(const MaterialX::VariableBlock&)>;

// Emits the per-material `<material_data_name>` struct (uniform inputs) and
// stack-data state that holds the BSDF graph evaluation state. Selection is
// driven by the requested layering mode plus the graph layering data attached
// to `context`. Verbatim port of `GenslangPtShaderGenerator::emit_material_data`.
void emit_material_data(
    const MaterialX::ShaderGenerator& shadergen,
    const EmitShaderInputsFn& emit_shader_inputs,
    const CalculateByteSizeFn& calculate_byte_size,
    const std::string& material_data_name,
    const LayeringDesc& layering,
    MaterialX::GenContext& context,
    MaterialX::ShaderStage& stage
);

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
