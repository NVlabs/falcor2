// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "codegen_types.h"

namespace falcor {
namespace materialx {

class FALCOR_API CodeGen {
public:
    static std::unique_ptr<CodeGenResult> generate(const CodeGenDesc& desc);
    static std::vector<std::string> discover_material_node_names(const CodeGenDesc& desc);
};

} // namespace materialx
} // namespace falcor
