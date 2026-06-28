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
namespace materialx {
namespace mx139 {
namespace emitter {

namespace mx = MaterialX;

namespace {
const std::string k_public_uniforms = mx::HW::PUBLIC_UNIFORMS;
}

void emit_material_data(
    const mx::ShaderGenerator& shadergen,
    const EmitShaderInputsFn& emit_shader_inputs,
    const CalculateByteSizeFn& calculate_byte_size,
    const std::string& material_data_name,
    const LayeringDesc& layering,
    mx::GenContext& context,
    mx::ShaderStage& stage
)
{
    auto user_data = context.getUserData<CodegenUserData>("mx139");
    mx::VariableBlock& public_uniforms = stage.getUniformBlock(k_public_uniforms);
    const ProfileDesc& profile = profile_desc();
    const ClosureRef root = layering.main_layer;
    const bool use_stack_frames = user_data->inputs.desc->layering_mode == LayeringMode::bsdf_mix && !root.is_none();

    shadergen.emitLine("public struct " + material_data_name, stage, false);
    shadergen.emitScopeBegin(stage);
    shadergen.emitLine("// Uniform graph inputs and texture handles.", stage, false);
    emit_shader_inputs(public_uniforms, stage, true, false);
    shadergen.emitScopeEnd(stage, true);
    shadergen.emitLineBreak(stage);

    user_data->inputs.result->data_struct_size = uint32_t(calculate_byte_size(public_uniforms));

    if (use_stack_frames)
        shadergen.emitLine("#define MX139_STACK_HAS_BSDF_FRAMES 1", stage, false);


    emit_basic_stack_data(profile.stack_data_type, layering, use_stack_frames, stage);
    shadergen.emitLineBreak(stage);
}

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
