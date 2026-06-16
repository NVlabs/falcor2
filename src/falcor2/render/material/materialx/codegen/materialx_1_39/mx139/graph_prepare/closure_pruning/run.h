// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"

#include <cstdint>

namespace MaterialX_v1_39_5 {
class ShaderGraph;
class GenContext;
} // namespace MaterialX_v1_39_5
namespace MaterialX = MaterialX_v1_39_5;

namespace falcor {
namespace materialx {
namespace mx139 {

class CodegenUserData;

FALCOR_API void
run_closure_pruning(MaterialX::ShaderGraph& graph, MaterialX::GenContext& context, CodegenUserData& user_data);

} // namespace mx139
} // namespace materialx
} // namespace falcor
