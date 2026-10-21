// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "codegen_types.h"

namespace falcor {
namespace mtlx {

class CodeGenCache;

class FALCOR_API CodeGen {
public:
    static std::unique_ptr<CodeGenResult> generate(const CodeGenDesc& desc, CodeGenCache* cache = nullptr);
    static std::vector<RenderableElement> discover_renderable_elements(const CodeGenDesc& desc);
};

} // namespace mtlx
} // namespace falcor
