// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "opacity_analysis.h"

#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderNode.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace mx = MaterialX;

namespace falcor {
namespace mtlx {
namespace codegen_support {

namespace {

std::optional<float> uniform_float(const ScalarStaticValue& value)
{
    if (value.kind == ScalarStaticValueKind::KnownZero)
        return 0.0f;
    if (value.kind == ScalarStaticValueKind::KnownOne)
        return 1.0f;
    if (value.kind != ScalarStaticValueKind::KnownConstant || value.value.size == 0)
        return {};

    const float result = value.value.values[0];
    if (!std::isfinite(result))
        return {};
    for (uint8_t i = 1; i < value.value.size; ++i) {
        if (!std::isfinite(value.value.values[i]) || value.value.values[i] != result)
            return {};
    }
    return result;
}

std::optional<float>
uniform_input_float(const mx::ShaderInput& input, const StaticInputQuery& static_input_query, float fallback)
{
    // The fallback values below are the schema defaults for the standard
    // scalar `mix` and `opacity` inputs. Custom surface nodes may use the same
    // input names with different types, which must remain dynamic.
    if (input.getType() != mx::Type::FLOAT)
        return {};

    const std::optional<float> value = uniform_float(static_input_query.input_value(input));
    if (value || input.getConnection())
        return value;
    return fallback;
}

class OpacityAnalyzer {
public:
    explicit OpacityAnalyzer(const StaticInputQuery& static_input_query)
        : m_static_input_query(static_input_query)
    {
    }

    std::optional<float> graph_opacity(const mx::ShaderGraph& graph)
    {
        if (graph.numOutputSockets() == 0)
            return {};
        const mx::ShaderGraphOutputSocket* output_socket = graph.getOutputSocket(0);
        return output_socket ? output_opacity(output_socket->getConnection()) : std::optional<float>{};
    }

private:
    std::optional<float> output_opacity(const mx::ShaderOutput* output)
    {
        if (!output || !m_active_outputs.insert(output).second)
            return {};

        const std::optional<float> result = node_opacity(*output);
        m_active_outputs.erase(output);
        return result;
    }

    std::optional<float> node_opacity(const mx::ShaderOutput& output)
    {
        const mx::ShaderNode* node = output.getNode();
        if (!node)
            return {};

        if (const mx::ShaderGraph* implementation_graph = node->getImplementation().getGraph()) {
            for (size_t i = 0; i < node->numOutputs(); ++i) {
                if (node->getOutput(i) != &output || i >= implementation_graph->numOutputSockets())
                    continue;
                const mx::ShaderGraphOutputSocket* inner_output = implementation_graph->getOutputSocket(i);
                return inner_output ? output_opacity(inner_output->getConnection()) : std::optional<float>{};
            }
            return {};
        }

        if (output.getType() == mx::Type::MATERIAL) {
            const mx::ShaderInput* surface = node->getInput(mx::ShaderNode::SURFACESHADER);
            return surface ? output_opacity(surface->getConnection()) : std::optional<float>{};
        }

        if (output.getType() != mx::Type::SURFACESHADER)
            return {};

        const mx::ShaderInput* foreground = node->getInput("fg");
        const mx::ShaderInput* background = node->getInput("bg");
        if (foreground && background) {
            const std::optional<float> foreground_opacity = output_opacity(foreground->getConnection());
            const std::optional<float> background_opacity = output_opacity(background->getConnection());
            const mx::ShaderInput* mix = node->getInput("mix");
            const std::optional<float> mix_value
                = mix ? uniform_input_float(*mix, m_static_input_query, 0.0f) : std::optional<float>{};
            if (!foreground_opacity || !background_opacity || !mix_value)
                return {};

            const float clamped_mix = std::clamp(*mix_value, 0.0f, 1.0f);
            return *background_opacity * (1.0f - clamped_mix) + *foreground_opacity * clamped_mix;
        }

        const mx::ShaderInput* opacity = node->getInput("opacity");
        if (!opacity)
            return 1.0f;
        const std::optional<float> value = uniform_input_float(*opacity, m_static_input_query, 1.0f);
        return value ? std::optional<float>{std::clamp(*value, 0.0f, 1.0f)} : std::optional<float>{};
    }

    const StaticInputQuery& m_static_input_query;
    std::unordered_set<const mx::ShaderOutput*> m_active_outputs;
};

} // namespace

std::optional<float>
materialx_graph_static_opacity(const mx::ShaderGraph& graph, const StaticInputQuery& static_input_query)
{
    return OpacityAnalyzer(static_input_query).graph_opacity(graph);
}

} // namespace codegen_support
} // namespace mtlx
} // namespace falcor
