// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "formal_node_operation.h"

#include <MaterialXGenShader/ShaderNode.h>

namespace falcor::mtlx::structured {

/// Create an ordinary source implementation that retains formal NodeDef identity.
MaterialX::ShaderNodeImplPtr create_source_code_node();

/// Return the formal operation retained by the structured source implementation.
/// Factory-registered implementations and compound graphs return `unknown`.
FormalNodeOperation formal_node_operation(const MaterialX::ShaderNode& node);

} // namespace falcor::mtlx::structured
