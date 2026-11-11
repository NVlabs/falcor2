// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"

#include <MaterialXGenShader/ShaderStage.h>

#include <string>
#include <string_view>

namespace falcor::mtlx::structured {

/// Emit a material-level writer that evaluates the existing Graph only when a registered output is enabled.
std::string emit_material_aov_writer(
    const MaterialX::VariableBlock& outputs,
    const CodeGenResult& result,
    std::string_view graph_name,
    std::string_view material_data_name
);

} // namespace falcor::mtlx::structured
