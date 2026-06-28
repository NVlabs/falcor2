// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "static_value_analysis.h"

#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderNode.h>

#include <string_view>

namespace falcor {
namespace materialx {
namespace mx139 {

// Read-only query facade over static values. Policy code should ask positive
// proof questions through this type instead of depending on analysis maps.
class StaticInputQuery {
public:
    StaticInputQuery() = default;
    StaticInputQuery(
        const OutputStaticValueMap& output_static_values,
        const MaterialX::VariableBlock* constants = nullptr
    );

    // Returns true only when the named input is proven exactly equal to value.
    // False means "not proven exact": it covers proven-different, unknown,
    // unsupported, non-finite, and hidden implementation-default values.
    bool is_exact(const MaterialX::ShaderNode& node, std::string_view input_name, float value) const;

private:
    ScalarStaticValue input_value(const MaterialX::ShaderInput& input) const;
    ScalarStaticValue constant_value(const MaterialX::ShaderInput& input) const;

    const OutputStaticValueMap* m_output_static_values = nullptr;
    const MaterialX::VariableBlock* m_constants = nullptr;
};

} // namespace mx139
} // namespace materialx
} // namespace falcor
