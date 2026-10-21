// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenShader/ShaderNodeImpl.h>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

/// Ray-tracing fallback for MaterialX heighttonormal. The inherited Slang
/// implementation uses screen-space derivatives; this implementation instead
/// resamples the upstream texture expression unless the request explicitly
/// enables Slang derivatives.
class HeightToNormalNode final : public mx::ShaderNodeImpl {
public:
    static mx::ShaderNodeImplPtr create();

    void
    emitFunctionDefinition(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
};

} // namespace falcor::mtlx::structured
