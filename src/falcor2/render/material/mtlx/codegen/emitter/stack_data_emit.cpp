// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "stack_data_emit.h"

#include "methods/basic/stack_data_basic_emit.h"

#include "../codegen_user_data.h"
#include "../profile_policy.h"

#include <MaterialXGenHw/HwConstants.h>
#include <MaterialXGenShader/HwShaderGenerator.h>

#include <cstdint>
#include <string>

namespace falcor {
namespace mtlx {
namespace emitter {

namespace mx = MaterialX;

namespace {
const std::string k_public_uniforms = mx::HW::PUBLIC_UNIFORMS;
}

void emit_material_data(
    const mx::ShaderGenerator& shadergen,
    const EmitShaderInputsFn& emit_shader_inputs,
    const CalculateDataSizeUpperBoundFn& calculate_data_size_upper_bound,
    const std::string& material_data_name,
    const LayeringDesc& layering,
    mx::GenContext& context,
    mx::ShaderStage& stage
)
{
    auto user_data = context.getUserData<CodegenUserData>("mtlx");
    mx::VariableBlock& public_uniforms = stage.getUniformBlock(k_public_uniforms);
    const ProfileDesc& profile = profile_desc();
    const ClosureRef root = layering.main_layer;
    const bool use_stack_frames = user_data->inputs.desc->layering_mode == LayeringMode::bsdf_mix && !root.is_none();

    shadergen.emitLine("public struct " + material_data_name, stage, false);
    shadergen.emitScopeBegin(stage);
    shadergen.emitLine("// Uniform graph inputs and texture handles.", stage, false);
    emit_shader_inputs(public_uniforms, stage, true, false);
    // Slang reflection assigns an empty struct size zero, but CUDA C++ assigns it a non-zero
    // size. Keep the generated type non-empty so reflected entry-point offsets match PTX.
    // This is ABI padding, not material data, so do not include it in data_struct_size below.
    if (public_uniforms.empty())
        shadergen.emitLine("public uint _empty_struct_padding", stage);
    shadergen.emitScopeEnd(stage, true);
    shadergen.emitLineBreak(stage);

    user_data->inputs.result->data_struct_size = calculate_data_size_upper_bound(public_uniforms);

    if (use_stack_frames)
        shadergen.emitLine("#define MTLX_STACK_HAS_BSDF_FRAMES 1", stage, false);


    emit_basic_stack_data(profile.stack_data_type, layering, use_stack_frames, stage);
    shadergen.emitLineBreak(stage);
}

} // namespace emitter
} // namespace mtlx
} // namespace falcor
