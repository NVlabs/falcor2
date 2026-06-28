// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenShader/ShaderNodeImpl.h>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace genslangpt {

namespace mx = MaterialX;

// MaterialX's standard Slang/GLSL heighttonormal implementation uses ddx/ddy,
// which is valid in raster/pixel shaders but not in our ray tracing material
// evaluation. This node keeps the same MaterialX surface by resampling the
// upstream height-producing texture expression at small UV offsets, so the
// path tracer gets structured normals instead of a flat fallback.
class HeightToNormalNode : public mx::ShaderNodeImpl {
public:
    static mx::ShaderNodeImplPtr create();

    void
    emitFunctionDefinition(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
};

} // namespace genslangpt
} // namespace mx139
} // namespace materialx
} // namespace falcor
