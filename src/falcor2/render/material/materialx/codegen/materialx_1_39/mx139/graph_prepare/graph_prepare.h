// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material/materialx/codegen/codegen_types.h"

#include <MaterialXCore/Document.h>

#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace graph_prepare {

void prepare_graph(
    MaterialX::DocumentPtr doc,
    MaterialX::ElementPtr root_element,
    const CodeGenDesc& desc,
    const std::string& target
);

} // namespace graph_prepare
} // namespace mx139
} // namespace materialx
} // namespace falcor
