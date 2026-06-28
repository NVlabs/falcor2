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

namespace falcor {
namespace materialx {
namespace mx139 {
namespace genslangpt {

namespace {

const char* const k_texcoord_input_name = "texcoord";

// Find the UV input that should receive each height sample offset. Texture
// nodes express this as `texcoord`, and intermediate nodes simply forward the
// suffixed outputs produced by the resampled subgraph.
const mx::ShaderInput* sample2d_input(const mx::ShaderNode& node)
{
    const mx::ShaderInput* input = node.getInput(k_texcoord_input_name);
    if (input && input->getType() == mx::Type::VECTOR2)
        return input;
    return nullptr;
}

bool contains_input(const std::vector<const mx::ShaderInput*>& inputs, const mx::ShaderInput* input)
{
    return std::find(inputs.begin(), inputs.end(), input) != inputs.end();
}

std::vector<std::string>
height_to_normal_sample_offset_strings(const std::string& sample_size_name, const std::string& offset_type_name)
{
    std::vector<std::string> offsets;
    offsets.reserve(9);
    for (int row = -1; row <= 1; ++row) {
        for (int col = -1; col <= 1; ++col) {
            offsets.emplace_back(fmt::format(" + {} * {}({}.0, {}.0)", sample_size_name, offset_type_name, col, row));
        }
    }
    return offsets;
}

// Ordered upstream expression slice used to recompute the height input. The
// output/input suffix lists let us emit the same MaterialX nodes nine times
// without mutating the source graph or colliding with the original variables.
struct TextureSampleSubgraph {
    std::vector<const mx::ShaderNode*> nodes;
    std::vector<const mx::ShaderInput*> connected_inputs;
    std::vector<const mx::ShaderInput*> sampling_inputs;
};

// Walk only the texture-expression inputs that feed the current heighttonormal
// node. Graph wrappers are traversal structure rather than executable nodes,
// and the current heighttonormal node must be excluded to avoid self-recursion.
void collect_texture_sample_subgraph(
    const mx::ShaderNode& node,
    const mx::ShaderNode& excluded_node,
    TextureSampleSubgraph& result,
    std::unordered_map<const mx::ShaderNode*, bool>& visited
)
{
    if (&node == &excluded_node || node.getUniqueId() == excluded_node.getUniqueId())
        return;
    if (node.isAGraph())
        return;
    if (visited[&node])
        return;
    visited[&node] = true;

    for (const mx::ShaderInput* input : node.getInputs()) {
        const mx::ShaderOutput* connection = input ? input->getConnection() : nullptr;
        const mx::ShaderNode* upstream_node = connection ? connection->getNode() : nullptr;
        if (upstream_node) {
            if (upstream_node == &excluded_node || upstream_node->getUniqueId() == excluded_node.getUniqueId())
                continue;
            if (upstream_node->isAGraph())
                continue;
            collect_texture_sample_subgraph(*upstream_node, excluded_node, result, visited);
            result.connected_inputs.push_back(input);
        }
    }

    result.nodes.push_back(&node);
    if (const mx::ShaderInput* sampling_input = sample2d_input(node))
        result.sampling_inputs.push_back(sampling_input);
}

TextureSampleSubgraph collect_texture_sample_subgraph(const mx::ShaderNode& node, const mx::ShaderNode& excluded_node)
{
    TextureSampleSubgraph result;
    std::unordered_map<const mx::ShaderNode*, bool> visited;
    collect_texture_sample_subgraph(node, excluded_node, result, visited);
    return result;
}

// Emit nine local copies of the upstream height expression. Only the actual
// sampling inputs get UV offsets; other connected inputs just point at the
// matching suffixed intermediate values for the current sample.
std::vector<std::string> emit_height_to_normal_resampled_subgraph(
    const mx::ShaderOutput& height_input_connection,
    const mx::ShaderOutput& output,
    const TextureSampleSubgraph& subgraph,
    const std::vector<std::string>& offsets,
    mx::GenContext& context,
    mx::ShaderStage& stage
)
{
    static constexpr size_t k_sample_count = 9;

    std::vector<std::string> samples;
    samples.reserve(k_sample_count);
    for (size_t sample_index = 0; sample_index < k_sample_count; ++sample_index) {
        const std::string output_suffix = fmt::format("_{}{}", output.getVariable(), sample_index);
        std::vector<const mx::ShaderOutput*> suffixed_outputs;

        for (const mx::ShaderNode* node : subgraph.nodes) {
            for (const mx::ShaderOutput* node_output : node->getOutputs()) {
                context.addOutputSuffix(node_output, output_suffix);
                suffixed_outputs.push_back(node_output);
            }
        }
        for (const mx::ShaderInput* connected_input : subgraph.connected_inputs) {
            const bool is_sampling_input = contains_input(subgraph.sampling_inputs, connected_input);
            context.addInputSuffix(
                connected_input,
                is_sampling_input ? output_suffix + offsets[sample_index] : output_suffix
            );
        }
        for (const mx::ShaderInput* sampling_input : subgraph.sampling_inputs) {
            if (!contains_input(subgraph.connected_inputs, sampling_input))
                context.addInputSuffix(sampling_input, offsets[sample_index]);
        }

        for (const mx::ShaderNode* node : subgraph.nodes)
            node->getImplementation().emitFunctionCall(*node, context, stage);

        for (const mx::ShaderInput* connected_input : subgraph.connected_inputs)
            context.removeInputSuffix(connected_input);
        for (const mx::ShaderInput* sampling_input : subgraph.sampling_inputs) {
            if (!contains_input(subgraph.connected_inputs, sampling_input))
                context.removeInputSuffix(sampling_input);
        }
        for (const mx::ShaderOutput* node_output : suffixed_outputs)
            context.removeOutputSuffix(node_output);

        samples.emplace_back(height_input_connection.getVariable() + output_suffix);
    }
    return samples;
}

// Build the height samples consumed by the Sobel reconstruction. If the input
// is not a texture-expression float chain, fall back to repeating the available
// value so the node still emits valid code for constant or unusual inputs.
std::vector<std::string>
height_to_normal_sample_strings(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage)
{
    static constexpr size_t k_sample_count = 9;
    static constexpr float k_filter_size = 1.0f;
    static constexpr float k_filter_offset = 0.0f;

    const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
    const mx::ShaderInput* in_input = node.getInput("in");
    FALCOR_CHECK(in_input, "MX139 heighttonormal node missing in input.");

    const mx::ShaderOutput* in_connection = in_input->getConnection();
    const mx::ShaderOutput* output = node.getOutput();
    FALCOR_CHECK(output, "MX139 heighttonormal node missing output.");

    std::vector<std::string> samples;

    if (in_connection && in_connection->getType() == mx::Type::FLOAT) {
        const mx::ShaderNode* upstream_node = in_connection->getNode();
        if (upstream_node) {
            const TextureSampleSubgraph subgraph = collect_texture_sample_subgraph(*upstream_node, node);
            if (!subgraph.sampling_inputs.empty()) {
                const std::string sample_input_value
                    = shadergen.getUpstreamResult(subgraph.sampling_inputs.front(), context);
                const std::string sample_size_name = output->getVariable() + "_sample_size";
                const std::string vector2_type = shadergen.getSyntax().getTypeName(mx::Type::VECTOR2);
                shadergen.emitLine(
                    vector2_type + " " + sample_size_name + " = mx_compute_sample_size_uv(" + sample_input_value + ", "
                        + fmt::format("{:.1f}", k_filter_size) + ", " + fmt::format("{:.1f}", k_filter_offset) + ")",
                    stage
                );

                const std::vector<std::string> offsets
                    = height_to_normal_sample_offset_strings(sample_size_name, vector2_type);
                samples = emit_height_to_normal_resampled_subgraph(
                    *in_connection,
                    *output,
                    subgraph,
                    offsets,
                    context,
                    stage
                );
            }
        }
    }

    if (samples.empty()) {
        FALCOR_CHECK(
            in_connection || in_input->getValue(),
            "MX139 heighttonormal node missing in connection or value."
        );
        const std::string value = shadergen.getUpstreamResult(in_input, context);
        samples = std::vector<std::string>(k_sample_count, value);
    }

    return samples;
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
    DEFINE_SHADER_STAGE(stage, mx::Stage::PIXEL)
    {
        const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
        shadergen.emitLine("// Fixed UV sample footprint for ray tracing shaders without ddx/ddy.", stage, false);
        shadergen.emitLine(
            "// The footprint is based on MaterialXGenMdl; upstream expression resampling is Falcor-specific.",
            stage,
            false
        );
        shadergen.emitLine(
            "float2 mx_compute_sample_size_uv(float2 uv, float filter_size, float filter_offset)",
            stage,
            false
        );
        shadergen.emitScopeBegin(stage);
        shadergen.emitLine("float2 deriv_uv_x = float2(0.001, 0.0) * 0.5", stage);
        shadergen.emitLine("float2 deriv_uv_y = float2(0.0, 0.001) * 0.5", stage);
        shadergen.emitLine("float deriv_x = abs(deriv_uv_x.x) + abs(deriv_uv_y.x)", stage);
        shadergen.emitLine("float deriv_y = abs(deriv_uv_x.y) + abs(deriv_uv_y.y)", stage);
        shadergen.emitLine("float sample_size_u = 2.0 * filter_size * deriv_x + filter_offset", stage);
        shadergen.emitLine("if (sample_size_u < 1.0e-5) sample_size_u = 1.0e-5", stage);
        shadergen.emitLine("float sample_size_v = 2.0 * filter_size * deriv_y + filter_offset", stage);
        shadergen.emitLine("if (sample_size_v < 1.0e-5) sample_size_v = 1.0e-5", stage);
        shadergen.emitLine("return float2(sample_size_u, sample_size_v)", stage);
        shadergen.emitScopeEnd(stage);
        shadergen.emitLineBreak(stage);

        shadergen.emitLine("float3 mx_normal_from_samples_sobel(float samples[9], float scale)", stage, false);
        shadergen.emitScopeBegin(stage);
        shadergen.emitLine(
            "float nx = samples[0] - samples[2] + (2.0 * samples[3]) - (2.0 * samples[5]) + samples[6] - "
            "samples[8]",
            stage
        );
        shadergen.emitLine(
            "float ny = samples[0] + (2.0 * samples[1]) + samples[2] - samples[6] - (2.0 * samples[7]) - "
            "samples[8]",
            stage
        );
        shadergen.emitLine("float3 n = normalize(float3(nx * scale, ny * scale, 0.125))", stage);
        shadergen.emitLine("return (n + 1.0) * 0.5", stage);
        shadergen.emitScopeEnd(stage);
        shadergen.emitLineBreak(stage);
    }
}

void HeightToNormalNode::emitFunctionCall(
    const mx::ShaderNode& node,
    mx::GenContext& context,
    mx::ShaderStage& stage
) const
{
    DEFINE_SHADER_STAGE(stage, mx::Stage::PIXEL)
    {
        const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
        const mx::ShaderInput* scale_input = node.getInput("scale");
        const mx::ShaderOutput* output = node.getOutput();
        FALCOR_CHECK(scale_input && output, "MX139 heighttonormal node missing scale input or output.");

        const std::vector<std::string> samples = height_to_normal_sample_strings(node, context, stage);
        FALCOR_CHECK(samples.size() == 9, "MX139 heighttonormal expected 9 samples.");

        const std::string sample_name = output->getVariable() + "_samples";
        shadergen.emitLineBegin(stage);
        shadergen.emitString("float " + sample_name + "[9] = {", stage);
        shadergen.emitLineEnd(stage, false);
        for (size_t i = 0; i < samples.size(); ++i) {
            shadergen.emitLineBegin(stage);
            shadergen.emitString("    " + samples[i], stage);
            if (i + 1 < samples.size())
                shadergen.emitString(",", stage);
            shadergen.emitLineEnd(stage, false);
        }
        shadergen.emitLine("}", stage);

        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, false, context, stage);
        shadergen.emitString(" = mx_normal_from_samples_sobel(" + sample_name + ", ", stage);
        shadergen.emitInput(scale_input, context, stage);
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage);
    }
}

} // namespace genslangpt
} // namespace mx139
} // namespace materialx
} // namespace falcor
