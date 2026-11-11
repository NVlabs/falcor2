// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <string>

namespace falcor::mtlx::structured {

class GenslangPtShaderGenerator;
class StaticInputQuery;

struct EmissionQuery {
    std::string predicate = "true";
    // Standalone numeric functions and query. The data type is substituted at module emission.
    std::string source;
};

/// Compile a conservative emission query that can run without a shading interaction.
/// Only proven state-independent expressions read current material data; unknown paths remain eligible.
EmissionQuery build_emission_query(
    const MaterialX::ShaderGraph& graph,
    const MaterialX::VariableBlock& uniforms,
    const StaticInputQuery& static_input_query,
    const GenslangPtShaderGenerator& generator,
    MaterialX::GenContext& context
);

} // namespace falcor::mtlx::structured
