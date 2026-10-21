// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "codegen.h"

#include "structured/document/loader.h"
#include "structured/generator.h"

namespace falcor::mtlx {

std::unique_ptr<CodeGenResult> CodeGen::generate(const CodeGenDesc& desc, CodeGenCache* cache)
{
    return structured::generate_internal(desc, cache);
}

std::vector<RenderableElement> CodeGen::discover_renderable_elements(const CodeGenDesc& desc)
{
    return structured::discover_renderable_elements(desc);
}

} // namespace falcor::mtlx
