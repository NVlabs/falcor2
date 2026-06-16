// SPDX-License-Identifier: Apache-2.0

#include "shader_input_value.h"

#include "falcor2/core/error.h"

#include <MaterialXCore/Value.h>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace codegen_support {

std::string required_shader_input_value_string(
    const mx::ShaderNode& node,
    const std::string& input_name,
    const std::string& fallback
)
{
    const mx::ShaderInput* input = node.getInput(input_name);
    FALCOR_CHECK(
        input,
        "MX139 node '{}' (impl '{}') is missing nodedef input '{}'.",
        node.getName(),
        node.getImplementation().getName(),
        input_name
    );
    if (!input->getValue())
        return fallback;
    return input->getValue()->getValueString();
}

std::string optional_shader_input_value_string(
    const mx::ShaderNode& node,
    const std::string& input_name,
    const std::string& fallback
)
{
    const mx::ShaderInput* input = node.getInput(input_name);
    if (!input || !input->getValue())
        return fallback;
    return input->getValue()->getValueString();
}

} // namespace codegen_support
} // namespace mx139
} // namespace materialx
} // namespace falcor
