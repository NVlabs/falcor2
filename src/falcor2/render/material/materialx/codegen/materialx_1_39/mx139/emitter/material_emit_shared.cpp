// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Shared router TU for the `material_emit` family. Hosts `emit_module_header`.

#include "material_emit.h"


#include "../codegen_support/snippet_loader.h"

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

void emit_module_header(
    const MaterialX::ShaderGenerator& shadergen,
    const std::string& module_name,
    const std::string& library_import,
    MaterialX::ShaderStage& stage
)
{
    std::string_view snippet_name = "module_header_basic.slang";
    std::string source = stage.getSourceCode();
    source += codegen_support::render_snippet(
        snippet_name,
        codegen_support::TokenMap{
            {"$MxModuleName", module_name},
            {"$MxLibraryImport", library_import},
        }
    );
    stage.setSourceCode(source);
    shadergen.emitLineBreak(stage);
}

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
