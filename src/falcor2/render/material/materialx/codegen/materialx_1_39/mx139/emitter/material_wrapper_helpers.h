// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../layering_analysis.h"

#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

// Read-only helpers shared by the material wrapper emitters. They translate
// layering analysis and graph output metadata into small Slang expressions, and
// never mutate the MaterialX graph or the LayeringDesc.
std::string root_lobes(const LayeringDesc& desc, ClosureRef ref);
bool root_transmissive(const LayeringDesc& desc, ClosureRef ref);
bool root_curve_scattering(const LayeringDesc& desc, ClosureRef ref);

// Return conservative wrapper expressions for graph outputs that may be either
// surface shader/material records or non-material values. These helpers keep the
// closure_tree, bsdf_mix, and method-specific wrapper paths aligned.
bool graph_output_can_emit(const MaterialX::VariableBlock& outputs);
std::string graph_emission_expr(const MaterialX::VariableBlock& outputs);
std::string graph_opacity_expr(const MaterialX::VariableBlock& outputs);
std::string graph_thin_walled_expr(const MaterialX::VariableBlock& outputs);

// Returns the stack initialization needed when graph opacity is represented as
// a synthetic root mix against SimpleBTDF. Each line is prefixed by
// `line_prefix` and terminated by a newline; returns empty when no setup is
// required.
std::string build_synthetic_opacity_stack_setup_text(
    const LayeringDesc& layering,
    const std::string& stack_name,
    const std::string& graph_opacity,
    bool use_stack_frames,
    const std::string& line_prefix
);

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
