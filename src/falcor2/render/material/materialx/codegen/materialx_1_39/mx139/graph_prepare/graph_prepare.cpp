// SPDX-License-Identifier: Apache-2.0

#include "graph_prepare.h"

#include "default_geomprop_materialization.h"

#include "falcor2/core/error.h"
#include "falcor2/core/macros.h"

#include <cstdint>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace graph_prepare {

void prepare_graph(
    MaterialX::DocumentPtr doc,
    MaterialX::ElementPtr root_element,
    const CodeGenDesc& desc,
    const std::string& target
)
{
    FALCOR_UNUSED(desc, target);

    const uint32_t default_geomprop_inputs_materialized
        = materialize_reachable_default_geomprop_inputs(doc, root_element);
    if (default_geomprop_inputs_materialized > 0) {
        std::string validation_message;
        if (!doc->validate(&validation_message))
            FALCOR_THROW("MaterialX 1.39 default geomprop materialization validation failed:\n{}", validation_message);
    }
}

} // namespace graph_prepare
} // namespace mx139
} // namespace materialx
} // namespace falcor
