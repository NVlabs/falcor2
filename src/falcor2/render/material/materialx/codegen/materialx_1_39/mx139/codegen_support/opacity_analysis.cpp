// SPDX-License-Identifier: Apache-2.0

#include "opacity_analysis.h"

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Types.h>
#include <MaterialXCore/Value.h>
#include <MaterialXGenShader/ShaderNode.h>

#include <algorithm>
#include <sstream>
#include <string>

namespace mx = MaterialX;

namespace falcor {
namespace materialx {
namespace mx139 {
namespace codegen_support {

namespace {

std::optional<float> value_to_float(const mx::ValuePtr& value)
{
    if (!value)
        return {};
    if (value->isA<float>())
        return value->asA<float>();
    if (value->isA<int>())
        return static_cast<float>(value->asA<int>());
    if (value->isA<bool>())
        return value->asA<bool>() ? 1.0f : 0.0f;
    std::stringstream stream(value->getValueString());
    float parsed = 0.0f;
    stream >> parsed;
    return stream.fail() ? std::optional<float>{} : std::optional<float>{parsed};
}

} // namespace

std::optional<float>
materialx_static_float_input_value(const mx::NodePtr& node, const std::string& input_name, float fallback)
{
    if (!node)
        return {};

    // Resolve in the same order MaterialX runtime resolution would: live
    // upstream connection -> Document Input literal -> nodedef default ->
    // C++ policy fallback. Note that an `interfacename=` binding without a
    // local value is NOT treated as "dynamic" here: at this codegen stage
    // we don't have the enclosing nodegraph instance context to resolve
    // the binding, and conservatively reporting "unknown" causes the
    // layering rewrite to keep transmissive sub-trees alive that would
    // otherwise prune. In practice for the analyses that call this helper
    // (opacity / mix), the nodedef default for the bound port is what we
    // want when the parent context is itself unauthored, and an authored
    // override would have produced a literal value or a real connection.
    if (mx::InputPtr input = node->getInput(input_name)) {
        if (input->getConnectedNode() || input->getConnectedOutput())
            return {};
        if (mx::ValuePtr value = input->getValue())
            return value_to_float(value);
    }
    if (mx::ConstNodeDefPtr decl = node->getNodeDef()) {
        if (mx::InputPtr decl_input = decl->getActiveInput(input_name)) {
            if (mx::ValuePtr value = decl_input->getValue())
                return value_to_float(value);
            return {};
        }
    }
    return fallback;
}

std::optional<float> materialx_node_static_opacity(const mx::NodePtr& node, int depth)
{
    if (!node || depth > 16)
        return {};

    if (node->getType() == mx::MATERIAL_TYPE_STRING)
        return materialx_node_static_opacity(node->getConnectedNode(mx::ShaderNode::SURFACESHADER), depth + 1);

    if (node->getType() != mx::SURFACE_SHADER_TYPE_STRING)
        return 1.0f;

    mx::InputPtr fg = node->getInput("fg");
    mx::InputPtr bg = node->getInput("bg");
    if (fg && bg) {
        mx::NodePtr fg_node = fg->getConnectedNode();
        mx::NodePtr bg_node = bg->getConnectedNode();
        if (!fg_node || !bg_node)
            return {};
        std::optional<float> fg_opacity = materialx_node_static_opacity(fg_node, depth + 1);
        std::optional<float> bg_opacity = materialx_node_static_opacity(bg_node, depth + 1);
        std::optional<float> mix_value = materialx_static_float_input_value(node, "mix", 0.0f);
        if (!fg_opacity || !bg_opacity || !mix_value)
            return {};
        const float clamped_mix = std::clamp(*mix_value, 0.0f, 1.0f);
        return *bg_opacity * (1.0f - clamped_mix) + *fg_opacity * clamped_mix;
    }

    return materialx_static_float_input_value(node, "opacity", 1.0f);
}

std::optional<float> materialx_element_static_opacity(const mx::ElementPtr& element)
{
    if (!element)
        return {};
    if (mx::NodePtr node = element->asA<mx::Node>())
        return materialx_node_static_opacity(node);
    if (mx::OutputPtr output = element->asA<mx::Output>())
        return materialx_node_static_opacity(output->getConnectedNode());
    return {};
}

bool materialx_element_opacity_is_statically_opaque(const mx::ElementPtr& element)
{
    std::optional<float> opacity = materialx_element_static_opacity(element);
    return opacity.has_value() && *opacity >= 1.0f;
}

} // namespace codegen_support
} // namespace mx139
} // namespace materialx
} // namespace falcor
