// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../../layering_analysis.h"

#include <MaterialXGenShader/ShaderStage.h>

#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

// Appends the basic-family `struct MxStackData { ... }` block to `stage`,
// rendered from `stack_data_basic.slang`. Trailing blank line is emitted by
// the caller. Built unconditionally (basic family is always present).
void emit_basic_stack_data(
    const std::string& stack_data_type,
    const LayeringDesc& layering,
    bool use_stack_frames,
    MaterialX::ShaderStage& stage
);

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
