// SPDX-License-Identifier: Apache-2.0

#include "codegen.h"

#include "materialx_1_39/codegen_wrapper_1_39.h"

namespace falcor {
namespace materialx {

std::unique_ptr<CodeGenResult> CodeGen::generate(const CodeGenDesc& desc)
{
    return materialx_1_39::CodeGen_1_39::generate(desc);
}

std::vector<std::string> CodeGen::discover_material_node_names(const CodeGenDesc& desc)
{
    return materialx_1_39::CodeGen_1_39::discover_material_node_names(desc);
}

} // namespace materialx
} // namespace falcor
