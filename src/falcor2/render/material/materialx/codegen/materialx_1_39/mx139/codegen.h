// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"

#include <MaterialXCore/Document.h>

#include <memory>

namespace falcor {
namespace materialx {
namespace mx139 {

using materialx::CodeGenDesc;
using materialx::CodeGenResult;

std::unique_ptr<CodeGenResult> generate_code(MaterialX::DocumentPtr doc, const CodeGenDesc& desc);

} // namespace mx139
} // namespace materialx
} // namespace falcor
