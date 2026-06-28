// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenShader/ShaderStage.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace codegen_support {

// Per-stage emission state for the snippet loader.
//
// Step 1: this struct is a placeholder owner for the emitted-snippet set.
// Existing call sites pass it through `ensure_snippet_emitted` but the
// emitted-set is not consulted yet -- behaviour stays identical to the
// pre-extraction `emit_rewrite_slang_snippet` to keep golden parity.
// Step 3 turns this into the real owner of the dedupe set.
struct SnippetEmitState {
    std::unordered_set<std::string> emitted_snippets;

    // Parked here in Step 1.3 (off the CodegenUserData god struct). The
    // deferred microfacet BSDF cleanup (Step 3) will move this onto the
    // node itself.
    bool direct_microfacet_helpers_emitted = false;
};

using TokenMap = std::unordered_map<std::string, std::string>;

// Loads a `.slang` snippet from MX139_SNIPPET_PATH, strips the file
// header + module marker, normalises line endings, and ensures a trailing
// newline. Cached across calls. Throws on read failure.
std::string load_snippet_source(std::string_view snippet_name);

// Stage 1 snippet rendering. Loads the snippet and applies token
// substitution (`$Mx<Alnum>`) via MaterialX's `tokenSubstitution`.
// Strictness rules (see Step 2 / "Stage 1 strictness rules" in the
// plan): every key in `tokens` must match `^\$Mx[A-Za-z0-9]+$`,
// and the rendered source must contain no leftover `$Mx<Alnum>+` token.
// Both violations throw.
std::string render_snippet(std::string_view snippet_name, const TokenMap& tokens = {});

// Appends `snippet_name` to `stage`'s source code.
// `emit_state` is reserved for the Step 3 dedupe owner; in Step 1 it is
// accepted but not consulted so output is byte-identical to the
// pre-extraction `emit_rewrite_slang_snippet`.
void ensure_snippet_emitted(MaterialX::ShaderStage& stage, SnippetEmitState& emit_state, std::string_view snippet_name);

} // namespace codegen_support
} // namespace mx139
} // namespace materialx
} // namespace falcor
