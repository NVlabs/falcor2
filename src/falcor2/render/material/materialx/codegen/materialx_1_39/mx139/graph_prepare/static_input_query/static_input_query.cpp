// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "static_input_query.h"

#include "input_static_values.h"

#include <MaterialXGenShader/ShaderNode.h>

#include <cmath>
#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {

namespace {

bool constant_is_exact(const ScalarStaticConstant& value, float target)
{
    if (value.size == 0 || !std::isfinite(target))
        return false;

    for (uint8_t i = 0; i < value.size; ++i) {
        if (!std::isfinite(value.values[i]) || value.values[i] != target)
            return false;
    }
    return true;
}

bool static_value_is_exact(const ScalarStaticValue& value, float target)
{
    if (!std::isfinite(target))
        return false;

    switch (value.kind) {
    case ScalarStaticValueKind::KnownZero:
        return target == 0.0f;
    case ScalarStaticValueKind::KnownOne:
        return target == 1.0f;
    case ScalarStaticValueKind::KnownConstant:
        return constant_is_exact(value.value, target);
    case ScalarStaticValueKind::Unknown:
        return false;
    }
    return false;
}

} // namespace

StaticInputQuery::StaticInputQuery(
    const OutputStaticValueMap& output_static_values,
    const MaterialX::VariableBlock* constants
)
    : m_output_static_values(&output_static_values)
    , m_constants(constants)
{
}

bool StaticInputQuery::is_exact(const MaterialX::ShaderNode& node, std::string_view input_name, float value) const
{
    const MaterialX::ShaderInput* input = node.getInput(std::string(input_name));
    if (!input)
        return false;

    return static_value_is_exact(input_value(*input), value);
}

ScalarStaticValue StaticInputQuery::input_value(const MaterialX::ShaderInput& input) const
{
    if (const MaterialX::ShaderOutput* upstream = input.getConnection()) {
        if (!m_output_static_values)
            return ScalarStaticValue::unknown();

        const auto it = m_output_static_values->find(upstream);
        return it != m_output_static_values->end() ? it->second.scalar : ScalarStaticValue::unknown();
    }

    // Bind inputs can expose implementation defaults that are not author-visible.
    // Static analysis may still prove their values through an upstream interface;
    // that case is handled by the connected-input branch above.
    if (input.isBindInput())
        return ScalarStaticValue::unknown();

    ScalarStaticValue value = lift_static_value(input.getValue());
    if (!is_unknown_static_value(value))
        return value;

    return constant_value(input);
}

ScalarStaticValue StaticInputQuery::constant_value(const MaterialX::ShaderInput& input) const
{
    if (!m_constants || input.getVariable().empty())
        return ScalarStaticValue::unknown();

    for (size_t i = 0; i < m_constants->size(); ++i) {
        const MaterialX::ShaderPort* constant = (*m_constants)[i];
        if (!constant || constant->getVariable() != input.getVariable())
            continue;

        return lift_static_value(constant->getValue());
    }
    return ScalarStaticValue::unknown();
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
