// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "codegen.h"

#include "materialx_1_39/codegen_wrapper_1_39.h"

namespace falcor {
namespace materialx {

std::unique_ptr<CodeGenResult> CodeGen::generate(const CodeGenDesc& desc)
{
    return materialx_1_39::CodeGen_1_39::generate(desc);
}

std::vector<RenderableElement> CodeGen::discover_renderable_elements(const CodeGenDesc& desc)
{
    return materialx_1_39::CodeGen_1_39::discover_renderable_elements(desc);
}

} // namespace materialx
} // namespace falcor
