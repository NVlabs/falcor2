// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXCore/Element.h>
#include <MaterialXCore/Node.h>

#include <optional>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace codegen_support {

// Returns the connected-input-free static value of `input_name`, or {} if the
// input is connected (i.e. value is not statically known) or could not be parsed.
std::optional<float>
materialx_static_float_input_value(const MaterialX::NodePtr& node, const std::string& input_name, float fallback);

std::optional<float> materialx_node_static_opacity(const MaterialX::NodePtr& node, int depth = 0);
std::optional<float> materialx_element_static_opacity(const MaterialX::ElementPtr& element);
bool materialx_element_opacity_is_statically_opaque(const MaterialX::ElementPtr& element);

} // namespace codegen_support
} // namespace mx139
} // namespace materialx
} // namespace falcor
