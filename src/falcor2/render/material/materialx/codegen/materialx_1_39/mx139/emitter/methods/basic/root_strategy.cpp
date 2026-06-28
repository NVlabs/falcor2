// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "root_strategy.h"

#include "bsdf_mix_emit.h"
#include "closure_tree_emit.h"

#include <map>
#include <vector>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

namespace {

std::string build_collect_extra_method_text(
    const LayeringDesc& layering,
    const std::string& line_prefix,
    const std::string& report_lines
)
{
    if (layering.main_layer.is_none())
        return {};

    std::string out;
    auto append = [&](const std::string& line)
    {
        out += line_prefix;
        out += line;
        out += '\n';
    };

    append("public override Optional<ExtraBsdfProperties>");
    append("collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)");
    append("{");
    append("    ExtraBsdfProperties result = {};");
    append("    int reported_count = 0;");
    append("    MxExtraBsdfPropertiesContext ctx = {};");
    append("    ctx.wi_node = shading_frame_ws.to_local(si.wi_ws);");
    append("    ctx.eval_scale = float3(1.0);");
    append("    ctx.node_frame_ws = shading_frame_ws;");
    append("    ctx.diagnostic_frame_ws = si.shading_frame_ws;");
    out += report_lines;
    append("    result.bsdf_count = uint(reported_count);");
    append("    return result;");
    append("}");
    return out;
}

class ClosureTreeExtraStableIndexBuilder {
public:
    explicit ClosureTreeExtraStableIndexBuilder(const LayeringDesc& layering)
        : m_layering(layering)
    {
    }

    std::vector<std::size_t> build()
    {
        append_ref(m_layering.main_layer);
        return m_stable_indices;
    }

private:
    void append_ref(ClosureRef ref)
    {
        if (ref.is_none())
            return;

        if (ref.is_bsdf()) {
            m_stable_indices.push_back(ref.bsdf_index());
            return;
        }

        append_combiner(m_layering.combiners[ref.combiner_index()]);
    }

    void append_combiner(const LayeringDesc::CombinerDesc& combiner)
    {
        const ClosureRef a_ref = combiner.children[0];
        const ClosureRef b_ref = combiner.children[1];

        switch (combiner.mode) {
        case CombinerMode::layer:
            append_ref(a_ref);
            append_ref(b_ref);
            return;
        case CombinerMode::mix:
            append_ref(b_ref);
            append_ref(a_ref);
            return;
        case CombinerMode::add:
            append_ref(a_ref);
            append_ref(b_ref);
            return;
        case CombinerMode::weighted_layer:
            FALCOR_UNREACHABLE();
        }

        FALCOR_UNREACHABLE();
    }

    const LayeringDesc& m_layering;
    std::vector<std::size_t> m_stable_indices;
};

std::string build_closure_tree_collect_extra_text(const LayeringDesc& layering, const std::string& line_prefix)
{
    if (layering.main_layer.is_none())
        return {};

    const std::vector<std::size_t> stable_indices = ClosureTreeExtraStableIndexBuilder(layering).build();

    std::string body_lines;
    auto append = [&](const std::string& line)
    {
        body_lines += line_prefix;
        body_lines += "    ";
        body_lines += line;
        body_lines += ";\n";
    };

    append("ctx.visit_index = 0");
    for (std::size_t encounter_index = 0; encounter_index < stable_indices.size(); ++encounter_index)
        append(
            "ctx.stable_indices[" + std::to_string(encounter_index)
            + "] = " + std::to_string(stable_indices[encounter_index])
        );
    append("bsdf.collect_extra_bsdf_properties(result, reported_count, ctx)");
    return body_lines;
}

class ClosureTreeRootStrategy : public RootStrategy {
public:
    void emit_root_types(
        const MaterialX::ShaderGenerator& /* shadergen */,
        const LayeringDesc& /* layering */,
        const ProfileDesc& /* profile */,
        const std::string& /* root_type_name */,
        MaterialX::ShaderStage& /* stage */
    ) const override
    {
    }

    CompositionAliases build_aliases(
        const LayeringDesc& layering,
        const ProfileDesc& profile,
        const std::string& aliases_name,
        const std::string& /* root_type_name */
    ) const override
    {
        return build_composition_aliases(layering, profile, aliases_name);
    }

    StackParams stack_params() const override { return StackParams{false}; }

    std::string emit_root_value(
        const MaterialX::ShaderGenerator& shadergen,
        const LayeringDesc& layering,
        const ProfileDesc& /* profile */,
        const std::string& /* root_type_name */,
        const std::string& stack_name,
        const std::string& /* wi_expr */,
        const CompositionAliases& aliases,
        MaterialX::ShaderStage& stage
    ) const override
    {
        std::map<ClosureRef, std::string> emitted_values;
        return emit_closure_tree_composition_value(
            shadergen,
            layering,
            aliases,
            layering.main_layer,
            stack_name,
            emitted_values,
            stage,
            true
        );
    }

    RootValueBlock build_root_value_text(
        const LayeringDesc& layering,
        const ProfileDesc& /* profile */,
        const std::string& /* root_type_name */,
        const std::string& stack_name,
        const std::string& /* wi_expr */,
        const CompositionAliases& aliases,
        const std::string& line_prefix
    ) const override
    {
        RootValueBlock block;
        std::map<ClosureRef, std::string> emitted_values;
        block.root_expr = build_closure_tree_composition_value_text(
            layering,
            aliases,
            layering.main_layer,
            stack_name,
            true,
            line_prefix,
            emitted_values,
            block.lines
        );
        return block;
    }

    std::string build_collect_extra_text(const LayeringDesc& layering, const std::string& line_prefix) const override
    {
        return build_collect_extra_method_text(
            layering,
            line_prefix,
            build_closure_tree_collect_extra_text(layering, line_prefix)
        );
    }
};

class BsdfMixRootStrategy : public RootStrategy {
public:
    void emit_root_types(
        const MaterialX::ShaderGenerator& shadergen,
        const LayeringDesc& layering,
        const ProfileDesc& profile,
        const std::string& root_type_name,
        MaterialX::ShaderStage& stage
    ) const override
    {
        emit_bsdf_mix_root_bsdf_type(shadergen, layering, profile, root_type_name, stage);
        shadergen.emitLineBreak(stage);
    }

    CompositionAliases build_aliases(
        const LayeringDesc& layering,
        const ProfileDesc& /* profile */,
        const std::string& aliases_name,
        const std::string& root_type_name
    ) const override
    {
        return build_bsdf_mix_composition_aliases(layering, aliases_name, root_type_name);
    }

    StackParams stack_params() const override { return StackParams{true}; }

    std::string emit_root_value(
        const MaterialX::ShaderGenerator& shadergen,
        const LayeringDesc& layering,
        const ProfileDesc& /* profile */,
        const std::string& /* root_type_name */,
        const std::string& stack_name,
        const std::string& wi_expr,
        const CompositionAliases& aliases,
        MaterialX::ShaderStage& stage
    ) const override
    {
        return emit_bsdf_mix_composition_value(shadergen, layering, aliases, stack_name, wi_expr, stage, true);
    }

    RootValueBlock build_root_value_text(
        const LayeringDesc& layering,
        const ProfileDesc& /* profile */,
        const std::string& /* root_type_name */,
        const std::string& stack_name,
        const std::string& wi_expr,
        const CompositionAliases& aliases,
        const std::string& line_prefix
    ) const override
    {
        RootValueBlock block;
        block.root_expr = build_bsdf_mix_composition_value_text(
            layering,
            aliases,
            stack_name,
            wi_expr,
            true,
            line_prefix,
            block.lines
        );
        return block;
    }

    std::string build_collect_extra_text(const LayeringDesc& layering, const std::string& line_prefix) const override
    {
        return build_collect_extra_method_text(
            layering,
            line_prefix,
            line_prefix + "    bsdf.collect_extra_bsdf_properties(result, reported_count, ctx);\n"
        );
    }
};

} // namespace

std::unique_ptr<RootStrategy> make_root_strategy(const LayeringDesc& layering, const CodeGenDesc& desc)
{
    const bool use_bsdf_mix = desc.layering_mode == LayeringMode::bsdf_mix && !layering.main_layer.is_none();
    if (use_bsdf_mix)
        return std::make_unique<BsdfMixRootStrategy>();
    return std::make_unique<ClosureTreeRootStrategy>();
}

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
