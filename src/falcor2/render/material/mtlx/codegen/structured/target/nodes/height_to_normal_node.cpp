// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "height_to_normal_node.h"

#include "falcor2/core/error.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <fmt/format.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace falcor::mtlx::structured {

namespace {

const mx::ShaderInput* sample2d_input(const mx::ShaderNode& node)
{
    const mx::ShaderInput* input = node.getInput("texcoord");
    return input && input->getType() == mx::Type::VECTOR2 ? input : nullptr;
}

bool contains(const std::vector<const mx::ShaderInput*>& inputs, const mx::ShaderInput* input)
{
    return std::find(inputs.begin(), inputs.end(), input) != inputs.end();
}

struct TextureSampleSubgraph {
    std::vector<const mx::ShaderNode*> nodes;
    std::vector<const mx::ShaderInput*> connected_inputs;
    std::vector<const mx::ShaderInput*> sampling_inputs;
};

void collect_texture_sample_subgraph(
    const mx::ShaderNode& node,
    const mx::ShaderNode& excluded,
    TextureSampleSubgraph& result,
    std::unordered_map<const mx::ShaderNode*, bool>& visited
)
{
    if (&node == &excluded || node.getUniqueId() == excluded.getUniqueId() || node.isAGraph() || visited[&node])
        return;
    visited[&node] = true;
    for (const mx::ShaderInput* input : node.getInputs()) {
        const mx::ShaderOutput* connection = input ? input->getConnection() : nullptr;
        const mx::ShaderNode* upstream = connection ? connection->getNode() : nullptr;
        if (!upstream || upstream == &excluded || upstream->getUniqueId() == excluded.getUniqueId()
            || upstream->isAGraph())
            continue;
        collect_texture_sample_subgraph(*upstream, excluded, result, visited);
        result.connected_inputs.push_back(input);
    }
    result.nodes.push_back(&node);
    if (const mx::ShaderInput* input = sample2d_input(node))
        result.sampling_inputs.push_back(input);
}

TextureSampleSubgraph collect_texture_sample_subgraph(const mx::ShaderNode& node, const mx::ShaderNode& excluded)
{
    TextureSampleSubgraph result;
    std::unordered_map<const mx::ShaderNode*, bool> visited;
    collect_texture_sample_subgraph(node, excluded, result, visited);
    return result;
}

std::vector<std::string> sample_offsets(const std::string& sample_size, const std::string& vector_type)
{
    std::vector<std::string> result;
    result.reserve(9);
    for (int row = -1; row <= 1; ++row) {
        for (int column = -1; column <= 1; ++column)
            result.push_back(fmt::format(" + {} * {}({}.0, {}.0)", sample_size, vector_type, column, row));
    }
    return result;
}

std::vector<std::string> emit_resampled_subgraph(
    const mx::ShaderOutput& height,
    const mx::ShaderOutput& output,
    const TextureSampleSubgraph& subgraph,
    const std::vector<std::string>& offsets,
    mx::GenContext& context,
    mx::ShaderStage& stage
)
{
    std::vector<std::string> samples;
    samples.reserve(9);
    for (std::size_t sample = 0; sample < 9; ++sample) {
        const std::string suffix = fmt::format("_{}{}", output.getVariable(), sample);
        std::vector<const mx::ShaderOutput*> suffixed_outputs;
        for (const mx::ShaderNode* node : subgraph.nodes) {
            for (const mx::ShaderOutput* node_output : node->getOutputs()) {
                context.addOutputSuffix(node_output, suffix);
                suffixed_outputs.push_back(node_output);
            }
        }
        for (const mx::ShaderInput* input : subgraph.connected_inputs)
            context.addInputSuffix(
                input,
                contains(subgraph.sampling_inputs, input) ? suffix + offsets[sample] : suffix
            );
        for (const mx::ShaderInput* input : subgraph.sampling_inputs) {
            if (!contains(subgraph.connected_inputs, input))
                context.addInputSuffix(input, offsets[sample]);
        }
        for (const mx::ShaderNode* node : subgraph.nodes)
            node->getImplementation().emitFunctionCall(*node, context, stage);
        for (const mx::ShaderInput* input : subgraph.connected_inputs)
            context.removeInputSuffix(input);
        for (const mx::ShaderInput* input : subgraph.sampling_inputs) {
            if (!contains(subgraph.connected_inputs, input))
                context.removeInputSuffix(input);
        }
        for (const mx::ShaderOutput* node_output : suffixed_outputs)
            context.removeOutputSuffix(node_output);
        samples.push_back(height.getVariable() + suffix);
    }
    return samples;
}

std::vector<std::string> height_samples(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage)
{
    const mx::ShaderGenerator& target = context.getShaderGenerator();
    const mx::ShaderInput* input = node.getInput("in");
    const mx::ShaderOutput* connection = input ? input->getConnection() : nullptr;
    const mx::ShaderOutput* output = node.getOutput();
    FALCOR_CHECK(input && output, "Structured MaterialX heighttonormal node is incomplete.");

    std::vector<std::string> result;
    const mx::ShaderNode* upstream
        = connection && connection->getType() == mx::Type::FLOAT ? connection->getNode() : nullptr;
    if (upstream) {
        const TextureSampleSubgraph subgraph = collect_texture_sample_subgraph(*upstream, node);
        if (!subgraph.sampling_inputs.empty()) {
            const std::string uv = target.getUpstreamResult(subgraph.sampling_inputs.front(), context);
            const std::string size = output->getVariable() + "_sample_size";
            const std::string vector_type = target.getSyntax().getTypeName(mx::Type::VECTOR2);
            target.emitLine(vector_type + " " + size + " = mx_compute_sample_size_uv(" + uv + ", 1.0, 0.0)", stage);
            result = emit_resampled_subgraph(
                *connection,
                *output,
                subgraph,
                sample_offsets(size, vector_type),
                context,
                stage
            );
        }
    }
    if (result.empty()) {
        FALCOR_CHECK(connection || input->getValue(), "Structured MaterialX heighttonormal input has no value.");
        result.assign(9, target.getUpstreamResult(input, context));
    }
    return result;
}

} // namespace

mx::ShaderNodeImplPtr HeightToNormalNode::create()
{
    return std::make_shared<HeightToNormalNode>();
}

void HeightToNormalNode::emitFunctionDefinition(
    const mx::ShaderNode&,
    mx::GenContext& context,
    mx::ShaderStage& stage
) const
{
    const mx::ShaderGenerator& target = context.getShaderGenerator();
    target.emitLine("// Fixed UV sample footprint for ray tracing shaders without ddx/ddy.", stage, false);
    target
        .emitLine("float2 mx_compute_sample_size_uv(float2 uv, float filter_size, float filter_offset)", stage, false);
    target.emitScopeBegin(stage);
    target.emitLine("float2 deriv_uv_x = float2(0.001, 0.0) * 0.5", stage);
    target.emitLine("float2 deriv_uv_y = float2(0.0, 0.001) * 0.5", stage);
    target.emitLine("float deriv_x = abs(deriv_uv_x.x) + abs(deriv_uv_y.x)", stage);
    target.emitLine("float deriv_y = abs(deriv_uv_x.y) + abs(deriv_uv_y.y)", stage);
    target.emitLine("float sample_size_u = max(2.0 * filter_size * deriv_x + filter_offset, 1.0e-5)", stage);
    target.emitLine("float sample_size_v = max(2.0 * filter_size * deriv_y + filter_offset, 1.0e-5)", stage);
    target.emitLine("return float2(sample_size_u, sample_size_v)", stage);
    target.emitScopeEnd(stage);
    target.emitLineBreak(stage);
    target.emitLine("float3 mx_normal_from_samples_sobel(float samples[9], float scale)", stage, false);
    target.emitScopeBegin(stage);
    target.emitLine(
        "float nx = samples[0] - samples[2] + 2.0 * samples[3] - 2.0 * samples[5] + samples[6] - samples[8]",
        stage
    );
    target.emitLine(
        "float ny = samples[0] + 2.0 * samples[1] + samples[2] - samples[6] - 2.0 * samples[7] - samples[8]",
        stage
    );
    target.emitLine("return (normalize(float3(nx * scale, ny * scale, 0.125)) + 1.0) * 0.5", stage);
    target.emitScopeEnd(stage);
    target.emitLineBreak(stage);
}

void HeightToNormalNode::emitFunctionCall(
    const mx::ShaderNode& node,
    mx::GenContext& context,
    mx::ShaderStage& stage
) const
{
    const mx::ShaderGenerator& target = context.getShaderGenerator();
    const mx::ShaderInput* scale = node.getInput("scale");
    const mx::ShaderOutput* output = node.getOutput();
    FALCOR_CHECK(scale && output, "Structured MaterialX heighttonormal node is missing scale or output.");
    const std::vector<std::string> samples = height_samples(node, context, stage);
    const std::string sample_name = output->getVariable() + "_samples";
    target.emitLineBegin(stage);
    target.emitString("float " + sample_name + "[9] = {", stage);
    target.emitLineEnd(stage, false);
    for (std::size_t i = 0; i < samples.size(); ++i)
        target.emitLine("    " + samples[i] + (i + 1 < samples.size() ? "," : ""), stage, false);
    target.emitLine("}", stage);
    target.emitLineBegin(stage);
    target.emitOutput(output, true, false, context, stage);
    target.emitString(" = mx_normal_from_samples_sobel(" + sample_name + ", ", stage);
    target.emitInput(scale, context, stage);
    target.emitString(")", stage);
    target.emitLineEnd(stage);
}

} // namespace falcor::mtlx::structured
