// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "bsdf_emission.h"

#include "../snippet_renderer.h"
#include "../user_data.h"

#include "falcor2/core/error.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

void emit_bsdf_preparation_helper(mx::GenContext& context, mx::ShaderStage& stage)
{
    auto user_data = context.getUserData<EmissionUserData>(USER_DATA_NAME);
    FALCOR_CHECK(user_data, "Structured MaterialX BSDF node has no generation state.");
    EmissionScratch& scratch = user_data->emission_scratch();
    if (scratch.bsdf_preparation_helper_emitted)
        return;
    scratch.bsdf_preparation_helper_emitted = true;

    // This shared asset predates the structured target and retains its
    // microfacet-oriented filename, but direct_prepare_bsdf() is the common
    // frame-preparation operation for every directly emitted BSDF leaf.
    std::string source = stage.getSourceCode();
    source += load_snippet("mx_direct_microfacet_helpers.slang");
    stage.setSourceCode(source);
}

void emit_bsdf_initialization(
    const mx::ShaderGenerator& target,
    mx::ShaderStage& stage,
    const std::string& output,
    const std::string& weight,
    const std::string& bsdf_expression,
    const std::string& normal,
    const std::string& tangent
)
{
    target.emitLine(output + ".set_bsdf(mx_stack_next_bsdf++)", stage);
    target.emitLine("#ifdef MTLX_SCALARIZED_GRAPH_STORAGE", stage, false);
    target.emitLine("mx_stack.set_bsdf_weight(" + output + ".get_index(), float3(" + weight + "))", stage);
    target.emitLine("#else", stage, false);
    target.emitLine("mx_stack.bsdf_weights[" + output + ".get_index()] = float3(" + weight + ")", stage);
    target.emitLine("#endif", stage, false);
    target.emitLine("mx_stack.set_bsdf(" + output + ".get_index(), " + bsdf_expression + ")", stage);
    target.emitLine("direct_prepare_bsdf(" + output + ", " + normal + ", " + tangent + ")", stage);
}

} // namespace falcor::mtlx::structured
