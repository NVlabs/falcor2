// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Shared signatures for the `emit_material_structs` / `emit_module_header`
// routers.

#pragma once

#include "../layering_analysis.h"

#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

// Callback used by the wrapper emitters for stable generated Slang snippets.
// Keeping it injected lets the snippet loader remain internal to the codegen
// translation unit until more emitters need a shared snippet API.
using EmitRewriteSnippetFn = std::function<void(MaterialX::ShaderStage&, std::string_view)>;

// Top-level material-wrapper emission request. Optional methods may consume the
// request before the basic emitters handle closure_tree/bsdf_mix.
struct MaterialStructsDesc {
    std::string material_name;
    std::string material_instance_name;
    std::string material_data_name;
    std::string graph_name;
    const LayeringDesc& layering;
    const CodeGenDesc& codegen_desc;
    bool graph_has_emission = false;
    std::size_t materialx_payload_size = 0;
};

void emit_material_structs(
    const MaterialX::ShaderGenerator& shadergen,
    const EmitRewriteSnippetFn& emit_rewrite_snippet,
    const MaterialStructsDesc& params,
    MaterialX::ShaderStage& stage
);

// Emits the per-module preamble: file-generated comment, `module` line, and
// import list. Followed by a trailing line break so the next `emit*` call can
// start a fresh line.
void emit_module_header(
    const MaterialX::ShaderGenerator& shadergen,
    const std::string& module_name,
    const std::string& library_import,
    MaterialX::ShaderStage& stage
);

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
