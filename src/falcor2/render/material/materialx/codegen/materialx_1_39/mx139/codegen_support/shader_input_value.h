// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenShader/ShaderNode.h>

#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace codegen_support {

namespace mx = MaterialX;

// Reads the literal value string of a ShaderInput on a ShaderNode.
//
// The MaterialX shader generator resolves nodedef defaults into the
// ShaderInput value before codegen runs, so this returns the post-resolution
// value. `fallback` is returned when the port exists but has no literal value
// (some string-typed inputs may legitimately be unpopulated by the loader).
// Asserts that the port is declared on the node's nodedef; intended for sites
// where the codegen contract guarantees the port exists (genslangpt builtin
// node impls and the geometry-stream discovery walk).
std::string required_shader_input_value_string(
    const mx::ShaderNode& node,
    const std::string& input_name,
    const std::string& fallback
);

// Same as required_shader_input_value_string but tolerates a missing port and
// returns `fallback` in that case. Intended for analyses that walk arbitrary
// closure nodes where the requested input may not be declared on every
// variant (e.g. reading `scatter_mode` on every BSDF in layering analysis).
std::string optional_shader_input_value_string(
    const mx::ShaderNode& node,
    const std::string& input_name,
    const std::string& fallback
);

} // namespace codegen_support
} // namespace mx139
} // namespace materialx
} // namespace falcor
