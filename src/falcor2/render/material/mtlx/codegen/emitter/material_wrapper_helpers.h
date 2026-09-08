// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../layering_analysis.h"

#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <string>

namespace falcor {
namespace mtlx {
namespace emitter {

// Read-only helpers shared by the material wrapper emitters. They translate
// layering analysis and graph output metadata into small Slang expressions, and
// never mutate the MaterialX graph or the LayeringDesc.
std::string root_lobes(const LayeringDesc& desc, ClosureRef ref);
bool root_transmissive(const LayeringDesc& desc, ClosureRef ref);

// Return conservative wrapper expressions for graph outputs that may be either
// surface shader/material records or non-material values. These helpers keep the
// closure_tree, bsdf_mix, and method-specific wrapper paths aligned.
bool graph_output_can_emit(const MaterialX::VariableBlock& outputs);
std::string graph_emission_expr(const MaterialX::VariableBlock& outputs);
std::string graph_opacity_expr(const MaterialX::VariableBlock& outputs);

} // namespace emitter
} // namespace mtlx
} // namespace falcor
