// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../../codegen_types.h"

#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace codegen_support {

// Postprocess transforms applied to generated Slang source.
// Keep the call order stable in the owning codegen pipeline; these helpers are
// pure byte-level transforms with no codegen-pipeline coupling.

void strip_slang_snippet_file_header(std::string& text);
void strip_slang_snippet_module_marker(std::string& text);

void replace_fresnel_factory_calls(std::string& source, const CodeGenDesc& desc);
void replace_compensation_factory_calls(std::string& source, const CodeGenDesc& desc);

void replace_hw_geom_tokens(std::string& source);
void mark_generated_void_methods_mutating(std::string& source);
void replace_glsl_compatibility_tokens(std::string& source);

} // namespace codegen_support
} // namespace mx139
} // namespace materialx
} // namespace falcor
