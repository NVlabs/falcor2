// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXCore/Types.h>
#include <MaterialXGenShader/ShaderNode.h>

#include <algorithm>
#include <string_view>

namespace falcor {
namespace materialx {
namespace mx139 {

inline bool string_starts_with(std::string_view s, std::string_view prefix)
{
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

inline std::string_view shader_node_implementation_name(const MaterialX::ShaderNode& node)
{
    return node.getImplementation().getName();
}

inline bool implementation_name_starts_with(std::string_view impl, std::string_view prefix)
{
    return string_starts_with(impl, prefix);
}

inline bool node_implementation_starts_with(const MaterialX::ShaderNode& node, std::string_view prefix)
{
    return implementation_name_starts_with(shader_node_implementation_name(node), prefix);
}

inline bool is_surface_closure_type(const MaterialX::TypeDesc& type)
{
    return type == MaterialX::Type::BSDF || type == MaterialX::Type::EDF;
}

inline bool is_any_closure_type(const MaterialX::TypeDesc& type)
{
    return is_surface_closure_type(type) || type == MaterialX::Type::VDF;
}

inline bool is_supported_weight_value_type(const MaterialX::TypeDesc& type)
{
    return type == MaterialX::Type::FLOAT || type == MaterialX::Type::COLOR3 || type == MaterialX::Type::COLOR4;
}

// Matches `IM_<op>_<closure>[_*]` for surface closure suffixes bsdf/edf.
// MX139 pruning handles VDF only through layer_vdf boundary policy.
inline bool matches_surface_closure_op(std::string_view impl, std::string_view op)
{
    constexpr std::string_view k_head = "IM_";
    if (impl.size() < k_head.size() + op.size() + 1 + 3)
        return false;
    if (impl.substr(0, k_head.size()) != k_head)
        return false;
    if (impl.substr(k_head.size(), op.size()) != op)
        return false;
    if (impl[k_head.size() + op.size()] != '_')
        return false;
    const std::string_view rest = impl.substr(k_head.size() + op.size() + 1);
    return string_starts_with(rest, "bsdf") || string_starts_with(rest, "edf");
}

inline bool is_layer_vdf_node(const MaterialX::ShaderNode& node)
{
    return node_implementation_starts_with(node, "IM_layer_vdf");
}

inline bool is_connected_layer_vdf_boundary(const MaterialX::ShaderNode& node)
{
    if (!is_layer_vdf_node(node))
        return false;
    const MaterialX::ShaderOutput* out = node.getOutput(0);
    if (!out || out->getType() != MaterialX::Type::BSDF)
        return false;
    const MaterialX::ShaderInput* base = node.getInput("base");
    return base && base->getType() == MaterialX::Type::VDF && base->getConnection() != nullptr;
}

inline bool is_weighted_layer_node(const MaterialX::ShaderNode& node)
{
    return node_implementation_starts_with(node, "IM_weighted_layer_");
}

inline bool is_constant_node(std::string_view impl)
{
    return implementation_name_starts_with(impl, "IM_constant_");
}

inline bool is_scalar_or_color_multiply_node(std::string_view impl)
{
    return implementation_name_starts_with(impl, "IM_multiply_float")
        || implementation_name_starts_with(impl, "IM_multiply_color");
}

inline bool is_invert_float_node(std::string_view impl)
{
    return implementation_name_starts_with(impl, "IM_invert_float");
}

inline bool is_dot_node(std::string_view impl)
{
    return implementation_name_starts_with(impl, "IM_dot_");
}

inline bool is_add_float_node(std::string_view impl)
{
    return implementation_name_starts_with(impl, "IM_add_float");
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
