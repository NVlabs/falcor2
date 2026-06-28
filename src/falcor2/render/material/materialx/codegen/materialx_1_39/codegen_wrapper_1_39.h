// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "../codegen_types.h"

#include <MaterialXCore/Document.h>

namespace falcor {
namespace materialx {
namespace materialx_1_39 {

using materialx::CodeGenDesc;
using materialx::CodeGenResult;

class FALCOR_API CodeGen_1_39 {
public:
    static MaterialX::DocumentPtr load_document(const CodeGenDesc& desc);
    static std::unique_ptr<CodeGenResult> generate(const CodeGenDesc& desc);
    static std::vector<RenderableElement> discover_renderable_elements(const CodeGenDesc& desc);
};

} // namespace materialx_1_39
} // namespace materialx
} // namespace falcor
