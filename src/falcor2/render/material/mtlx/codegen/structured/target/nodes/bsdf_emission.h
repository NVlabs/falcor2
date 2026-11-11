// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <string>

namespace falcor::mtlx::structured {

/// Emit the shared frame-preparation helper required by structured BSDF nodes.
void emit_bsdf_preparation_helper(MaterialX::GenContext& context, MaterialX::ShaderStage& stage);

/// Emit the mechanical graph-storage protocol around an already-resolved,
/// concrete BSDF construction expression.
void emit_bsdf_initialization(
    const MaterialX::ShaderGenerator& target,
    MaterialX::ShaderStage& stage,
    const std::string& output,
    const std::string& weight,
    const std::string& bsdf_expression,
    const std::string& normal,
    const std::string& tangent
);

} // namespace falcor::mtlx::structured
