// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_wrapper_helpers.h"

#include "falcor2/core/error.h"

#include <MaterialXCore/Types.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <optional>
#include <set>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

namespace {

std::string lobe_or(const std::string& a, const std::string& b)
{
    if (a.empty())
        return b;
    if (b.empty())
        return a;
    return "(" + a + " | " + b + ")";
}

void collect_root_lobes(const LayeringDesc& desc, ClosureRef ref, std::set<std::string>& lobes)
{
    if (ref.is_none())
        return;

    if (ref.is_bsdf()) {
        const std::string& bsdf_lobes = desc.bsdfs[ref.bsdf_index()].lobe_types;
        static const std::array<std::string, 6> k_lobe_tokens = {
            "LobeTypes::diffuse_reflection",
            "LobeTypes::diffuse_transmission",
            "LobeTypes::glossy_reflection",
            "LobeTypes::glossy_transmission",
            "LobeTypes::glossy_curve",
            "LobeTypes::delta_transmission",
        };
        for (const std::string& token : k_lobe_tokens) {
            if (bsdf_lobes.find(token) != std::string::npos)
                lobes.insert(token);
        }
        return;
    }

    const LayeringDesc::CombinerDesc& combiner = desc.combiners[ref.combiner_index()];
    for (ClosureRef child : combiner.children)
        collect_root_lobes(desc, child, lobes);
}

} // namespace

std::string root_lobes(const LayeringDesc& desc, ClosureRef ref)
{
    std::set<std::string> lobes;
    collect_root_lobes(desc, ref, lobes);
    std::string result;
    for (const std::string& lobe : lobes)
        result = lobe_or(result, lobe);
    return result;
}

bool root_transmissive(const LayeringDesc& desc, ClosureRef ref)
{
    if (ref.is_none())
        return false;

    if (ref.is_bsdf())
        return desc.bsdfs[ref.bsdf_index()].transmissive;
    const LayeringDesc::CombinerDesc& combiner = desc.combiners[ref.combiner_index()];
    return std::any_of(
        combiner.children.begin(),
        combiner.children.end(),
        [&](ClosureRef child)
        {
            return root_transmissive(desc, child);
        }
    );
}

bool root_curve_scattering(const LayeringDesc& desc, ClosureRef ref)
{
    if (ref.is_none())
        return false;

    if (ref.is_bsdf())
        return desc.bsdfs[ref.bsdf_index()].curve_scattering;
    const LayeringDesc::CombinerDesc& combiner = desc.combiners[ref.combiner_index()];
    return std::any_of(
        combiner.children.begin(),
        combiner.children.end(),
        [&](ClosureRef child)
        {
            return root_curve_scattering(desc, child);
        }
    );
}

bool graph_output_can_emit(const MaterialX::VariableBlock& outputs)
{
    if (outputs.empty())
        return false;
    const MaterialX::TypeDesc type = outputs[0]->getType();
    return type == MaterialX::Type::SURFACESHADER || type == MaterialX::Type::MATERIAL;
}

std::string graph_emission_expr(const MaterialX::VariableBlock& outputs)
{
    if (!graph_output_can_emit(outputs))
        return "float3(0.0)";
    return "mtlx_graph." + outputs[0]->getVariable() + ".edf.radiance";
}

std::string graph_opacity_expr(const MaterialX::VariableBlock& outputs)
{
    if (!graph_output_can_emit(outputs))
        return "1.0";
    return "clamp(mtlx_graph." + outputs[0]->getVariable() + ".opacity, 0.0, 1.0)";
}

std::string graph_thin_walled_expr(const MaterialX::VariableBlock& outputs)
{
    if (!graph_output_can_emit(outputs))
        return "false";
    return "mtlx_graph." + outputs[0]->getVariable() + ".thin_walled";
}

void emit_synthetic_opacity_stack_setup(
    const MaterialX::ShaderGenerator& shadergen,
    const LayeringDesc& layering,
    const std::string& stack_name,
    const std::string& graph_opacity,
    bool use_stack_frames,
    MaterialX::ShaderStage& stage
)
{
    const std::optional<size_t> transparent_bsdf = synthetic_opacity_bsdf_index(layering);
    if (!transparent_bsdf)
        return;

    const std::optional<size_t> opacity_mix = synthetic_opacity_combiner_index(layering);
    FALCOR_CHECK(opacity_mix.has_value(), "Synthetic Mx139 opacity mix is missing its root mix combiner.");

    shadergen
        .emitLine("// Stochastic root opacity is represented as mix(actual_root, SimpleBTDF, opacity).", stage, false);
    shadergen.emitLine(fmt::format("{}.set_bsdf({}, MxSimpleBTDF())", stack_name, *transparent_bsdf), stage);
    shadergen.emitLine(fmt::format("{}.bsdf_weights[{}] = float3(1.0)", stack_name, *transparent_bsdf), stage);
    shadergen.emitLine(
        fmt::format("{}.bsdf_transmission_scales[{}] = float3(1.0)", stack_name, *transparent_bsdf),
        stage
    );
    shadergen.emitLine(fmt::format("{}.bsdf_n[{}] = float3(0.0, 0.0, 1.0)", stack_name, *transparent_bsdf), stage);
    shadergen.emitLine(fmt::format("{}.bsdf_t[{}] = float3(1.0, 0.0, 0.0)", stack_name, *transparent_bsdf), stage);
    if (use_stack_frames) {
        shadergen.emitLine(
            fmt::format(
                "{}.bsdf_frames[{}] = Frame({}.bsdf_n[{}], {}.bsdf_t[{}])",
                stack_name,
                *transparent_bsdf,
                stack_name,
                *transparent_bsdf,
                stack_name,
                *transparent_bsdf
            ),
            stage
        );
    }
    shadergen.emitLine(fmt::format("{}.layer_weights[{}] = float3(1.0)", stack_name, *opacity_mix), stage);
    shadergen.emitLine(fmt::format("{}.layer_transmission_scales[{}] = float3(1.0)", stack_name, *opacity_mix), stage);
    shadergen.emitLine(fmt::format("{}.mix_values[{}] = float3({})", stack_name, *opacity_mix, graph_opacity), stage);
}

std::string build_synthetic_opacity_stack_setup_text(
    const LayeringDesc& layering,
    const std::string& stack_name,
    const std::string& graph_opacity,
    bool use_stack_frames,
    const std::string& line_prefix
)
{
    const std::optional<size_t> transparent_bsdf = synthetic_opacity_bsdf_index(layering);
    if (!transparent_bsdf)
        return {};

    const std::optional<size_t> opacity_mix = synthetic_opacity_combiner_index(layering);
    FALCOR_CHECK(opacity_mix.has_value(), "Synthetic Mx139 opacity mix is missing its root mix combiner.");

    std::string out;
    auto append_line = [&](const std::string& text, bool semi = true)
    {
        out += line_prefix;
        out += text;
        if (semi)
            out += ';';
        out += '\n';
    };

    append_line("// Stochastic root opacity is represented as mix(actual_root, SimpleBTDF, opacity).", false);
    append_line(fmt::format("{}.set_bsdf({}, MxSimpleBTDF())", stack_name, *transparent_bsdf));
    append_line(fmt::format("{}.bsdf_weights[{}] = float3(1.0)", stack_name, *transparent_bsdf));
    append_line(fmt::format("{}.bsdf_transmission_scales[{}] = float3(1.0)", stack_name, *transparent_bsdf));
    append_line(fmt::format("{}.bsdf_n[{}] = float3(0.0, 0.0, 1.0)", stack_name, *transparent_bsdf));
    append_line(fmt::format("{}.bsdf_t[{}] = float3(1.0, 0.0, 0.0)", stack_name, *transparent_bsdf));
    if (use_stack_frames) {
        append_line(
            fmt::format(
                "{}.bsdf_frames[{}] = Frame({}.bsdf_n[{}], {}.bsdf_t[{}])",
                stack_name,
                *transparent_bsdf,
                stack_name,
                *transparent_bsdf,
                stack_name,
                *transparent_bsdf
            )
        );
    }
    append_line(fmt::format("{}.layer_weights[{}] = float3(1.0)", stack_name, *opacity_mix));
    append_line(fmt::format("{}.layer_transmission_scales[{}] = float3(1.0)", stack_name, *opacity_mix));
    append_line(fmt::format("{}.mix_values[{}] = float3({})", stack_name, *opacity_mix, graph_opacity));
    return out;
}

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
