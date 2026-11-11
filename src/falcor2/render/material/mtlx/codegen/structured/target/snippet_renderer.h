// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>

namespace falcor::mtlx::structured {

using SnippetTokens = std::unordered_map<std::string, std::string>;

/// Load one permanent shader asset relative to the configured snippet root.
///
/// Structured owns loading and substitution of the permanent .slang assets;
/// no other C++ codegen service participates in this path.
std::string load_snippet(std::string_view name);
/// Load a shader asset and replace its declared code-generation tokens.
std::string render_snippet(std::string_view name, const SnippetTokens& tokens);

} // namespace falcor::mtlx::structured
