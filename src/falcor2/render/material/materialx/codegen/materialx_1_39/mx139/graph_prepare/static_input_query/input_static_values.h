// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "static_value_analysis.h"

#include <MaterialXCore/Value.h>
#include <MaterialXGenShader/ShaderNode.h>

namespace falcor {
namespace materialx {
namespace mx139 {

namespace input_static_values_detail {

inline bool all_components_equal(const float& v, float target)
{
    return v == target;
}

inline bool all_components_equal(const MaterialX::Color3& v, float target)
{
    return v[0] == target && v[1] == target && v[2] == target;
}

inline bool all_components_equal(const MaterialX::Color4& v, float target)
{
    return v[0] == target && v[1] == target && v[2] == target && v[3] == target;
}

inline bool all_components_equal(const MaterialX::Vector2& v, float target)
{
    return v[0] == target && v[1] == target;
}

inline bool all_components_equal(const MaterialX::Vector3& v, float target)
{
    return v[0] == target && v[1] == target && v[2] == target;
}

inline bool all_components_equal(const MaterialX::Vector4& v, float target)
{
    return v[0] == target && v[1] == target && v[2] == target && v[3] == target;
}

inline void pack_scalar_static_constant(const float& v, ScalarStaticConstant& out)
{
    out.values = {v, 0, 0, 0};
    out.size = 1;
}

inline void pack_scalar_static_constant(const MaterialX::Color3& v, ScalarStaticConstant& out)
{
    out.values = {v[0], v[1], v[2], 0};
    out.size = 3;
}

inline void pack_scalar_static_constant(const MaterialX::Color4& v, ScalarStaticConstant& out)
{
    out.values = {v[0], v[1], v[2], v[3]};
    out.size = 4;
}

inline void pack_scalar_static_constant(const MaterialX::Vector2& v, ScalarStaticConstant& out)
{
    out.values = {v[0], v[1], 0, 0};
    out.size = 2;
}

inline void pack_scalar_static_constant(const MaterialX::Vector3& v, ScalarStaticConstant& out)
{
    out.values = {v[0], v[1], v[2], 0};
    out.size = 3;
}

inline void pack_scalar_static_constant(const MaterialX::Vector4& v, ScalarStaticConstant& out)
{
    out.values = {v[0], v[1], v[2], v[3]};
    out.size = 4;
}

template<typename T>
inline ScalarStaticValue lift_typed_static_value(const MaterialX::Value& value)
{
    if (!value.isA<T>())
        return ScalarStaticValue::unknown();

    const T& data = value.asA<T>();
    if (all_components_equal(data, 0.0f))
        return ScalarStaticValue::zero();
    if (all_components_equal(data, 1.0f))
        return ScalarStaticValue::one();

    ScalarStaticConstant constant;
    pack_scalar_static_constant(data, constant);
    return ScalarStaticValue::constant(constant);
}

} // namespace input_static_values_detail

inline ScalarStaticValue lift_static_value(const MaterialX::ValuePtr& value)
{
    if (!value)
        return ScalarStaticValue::unknown();
    if (value->isA<float>())
        return input_static_values_detail::lift_typed_static_value<float>(*value);
    if (value->isA<MaterialX::Color3>())
        return input_static_values_detail::lift_typed_static_value<MaterialX::Color3>(*value);
    if (value->isA<MaterialX::Color4>())
        return input_static_values_detail::lift_typed_static_value<MaterialX::Color4>(*value);
    if (value->isA<MaterialX::Vector2>())
        return input_static_values_detail::lift_typed_static_value<MaterialX::Vector2>(*value);
    if (value->isA<MaterialX::Vector3>())
        return input_static_values_detail::lift_typed_static_value<MaterialX::Vector3>(*value);
    if (value->isA<MaterialX::Vector4>())
        return input_static_values_detail::lift_typed_static_value<MaterialX::Vector4>(*value);
    return ScalarStaticValue::unknown();
}

inline OutputStaticValue
output_static_value_or_unknown(const MaterialX::ShaderOutput* output, const OutputStaticValueMap& current)
{
    if (!output)
        return OutputStaticValue{ClosureStaticValue::Unknown, ScalarStaticValue::unknown()};
    const auto it = current.find(output);
    return (it != current.end()) ? it->second
                                 : OutputStaticValue{ClosureStaticValue::Unknown, ScalarStaticValue::unknown()};
}

inline OutputStaticValue input_static_value(const MaterialX::ShaderInput& input, const OutputStaticValueMap& current)
{
    if (const MaterialX::ShaderOutput* upstream = input.getConnection())
        return output_static_value_or_unknown(upstream, current);

    OutputStaticValue value;
    value.scalar = lift_static_value(input.getValue());
    value.closure = ClosureStaticValue::Unknown;
    return value;
}

inline OutputStaticValue socket_default_static_value(const MaterialX::ShaderOutput* socket)
{
    OutputStaticValue value;
    value.scalar = socket ? lift_static_value(socket->getValue()) : ScalarStaticValue::unknown();
    value.closure = ClosureStaticValue::Unknown;
    return value;
}

inline ScalarStaticValue scalar_of(const MaterialX::ShaderInput& input, const OutputStaticValueMap& current)
{
    return input_static_value(input, current).scalar;
}

inline ScalarStaticValue scalar_of(const MaterialX::ShaderInput* input, const OutputStaticValueMap& current)
{
    return input ? scalar_of(*input, current) : ScalarStaticValue::unknown();
}

inline ClosureStaticValue closure_of(const MaterialX::ShaderInput* input, const OutputStaticValueMap& current)
{
    if (!input)
        return ClosureStaticValue::KnownEmpty;
    if (const MaterialX::ShaderOutput* upstream = input->getConnection())
        return output_static_value_or_unknown(upstream, current).closure;

    return ClosureStaticValue::KnownEmpty;
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
