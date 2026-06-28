// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "closure_tree_emit.h"

#include "root_strategy.h"
#include "../../material_wrapper_helpers.h"
#include "../../../codegen_support/snippet_loader.h"

#include <fmt/format.h>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

namespace {

const std::string k_outputs = "o";

} // namespace

void emit_closure_tree_composition_aliases(
    const MaterialX::ShaderGenerator& shadergen,
    const CompositionAliases& aliases,
    const ProfileDesc& profile,
    MaterialX::ShaderStage& stage
)
{
    shadergen.emitLine("public struct " + aliases.namespace_name, stage, false);
    shadergen.emitScopeBegin(stage);
    if (aliases.uses_weighted_helpers) {
        shadergen.emitLine(
            "// Generated WeightedLayer<N> aliases below mean Weighted<Layer<N>> traversal wrappers.",
            stage,
            false
        );
        shadergen.emitLine(
            "typealias Weighted<T : ILayerableBSDF> = " + profile.helper_type("WeightedBSDF") + "<T>",
            stage
        );
        shadergen.emitLine(
            "typealias FramedWeighted<T : ILayerableBSDF> = " + profile.helper_type("FramedWeightedBSDF") + "<T>",
            stage
        );
    }
    for (const CompositionAliases::AliasDesc& alias : aliases.aliases)
        shadergen.emitLine("typealias " + alias.name + " = " + alias.type, stage);
    shadergen.emitScopeEnd(stage, true);
    shadergen.emitLineBreak(stage);
}

std::string emit_closure_tree_composition_value(
    const MaterialX::ShaderGenerator& shadergen,
    const LayeringDesc& layering,
    const CompositionAliases& aliases,
    ClosureRef ref,
    const std::string& stack_name,
    std::map<ClosureRef, std::string>& emitted_values,
    MaterialX::ShaderStage& stage,
    bool use_local_aliases
)
{
    auto it = emitted_values.find(ref);
    if (it != emitted_values.end())
        return it->second;

    if (ref.is_none())
        return alias_reference(aliases, "RootBSDF", use_local_aliases) + "()";

    if (ref.is_bsdf()) {
        const std::size_t bsdf_index = ref.bsdf_index();
        const std::string value_name = fmt::format("bsdf_{}", bsdf_index);
        const std::string frame_expr = layering.bsdfs[bsdf_index].preserve_tangent_frame
            ? fmt::format(
                  "Frame({}.bsdf_n[{}], {}.bsdf_t[{}], 1.0, Frame::Component::tangent)",
                  stack_name,
                  bsdf_index,
                  stack_name,
                  bsdf_index
              )
            : fmt::format("Frame({}.bsdf_n[{}], {}.bsdf_t[{}])", stack_name, bsdf_index, stack_name, bsdf_index);
        shadergen.emitLine(
            fmt::format(
                "let {} = {}({}.bsdf{}, {}.bsdf_weights[{}], {}.bsdf_transmission_scales[{}], {})",
                value_name,
                alias_reference(aliases, aliases.weighted_aliases.at(ref), use_local_aliases),
                stack_name,
                bsdf_index,
                stack_name,
                bsdf_index,
                stack_name,
                bsdf_index,
                frame_expr
            ),
            stage
        );
        emitted_values[ref] = value_name;
        return value_name;
    }

    const std::size_t combiner_index = ref.combiner_index();
    const LayeringDesc::CombinerDesc& combiner = layering.combiners[combiner_index];
    const ClosureRef a_ref = combiner.children[0];
    const ClosureRef b_ref = combiner.children[1];
    const std::string a = emit_closure_tree_composition_value(
        shadergen,
        layering,
        aliases,
        a_ref,
        stack_name,
        emitted_values,
        stage,
        use_local_aliases
    );
    const std::string b = emit_closure_tree_composition_value(
        shadergen,
        layering,
        aliases,
        b_ref,
        stack_name,
        emitted_values,
        stage,
        use_local_aliases
    );
    const std::string core_name = fmt::format("{}_{}", combiner_value_prefix(combiner.mode), combiner_index);
    const std::string weighted_name = fmt::format("{}_weighted", core_name);

    switch (combiner.mode) {
    case CombinerMode::layer:
        shadergen.emitLine(
            fmt::format(
                "let {} = {}({}, {})",
                core_name,
                alias_reference(aliases, aliases.core_aliases.at(ref), use_local_aliases),
                a,
                b
            ),
            stage
        );
        break;
    case CombinerMode::mix:
        shadergen.emitLine(
            fmt::format(
                "let {} = {}({}, {}, {}.mix_values[{}])",
                core_name,
                alias_reference(aliases, aliases.core_aliases.at(ref), use_local_aliases),
                a,
                b,
                stack_name,
                combiner_index
            ),
            stage
        );
        break;
    case CombinerMode::add:
        shadergen.emitLine(
            fmt::format(
                "let {} = {}({}, {})",
                core_name,
                alias_reference(aliases, aliases.core_aliases.at(ref), use_local_aliases),
                a,
                b
            ),
            stage
        );
        break;
    case CombinerMode::weighted_layer:
        FALCOR_UNREACHABLE();
    }

    shadergen.emitLine(
        fmt::format(
            "let {} = {}({}, {}.layer_weights[{}], {}.layer_transmission_scales[{}])",
            weighted_name,
            alias_reference(aliases, aliases.weighted_aliases.at(ref), use_local_aliases),
            core_name,
            stack_name,
            combiner_index,
            stack_name,
            combiner_index
        ),
        stage
    );
    emitted_values[ref] = weighted_name;
    return weighted_name;
}

std::string build_closure_tree_composition_value_text(
    const LayeringDesc& layering,
    const CompositionAliases& aliases,
    ClosureRef ref,
    const std::string& stack_name,
    bool use_local_aliases,
    const std::string& line_prefix,
    std::map<ClosureRef, std::string>& emitted_values,
    std::string& out
)
{
    auto it = emitted_values.find(ref);
    if (it != emitted_values.end())
        return it->second;

    auto append_line = [&](const std::string& text)
    {
        out += line_prefix;
        out += text;
        out += ";\n";
    };

    if (ref.is_none())
        return alias_reference(aliases, "RootBSDF", use_local_aliases) + "()";

    if (ref.is_bsdf()) {
        const std::size_t bsdf_index = ref.bsdf_index();
        const std::string value_name = fmt::format("bsdf_{}", bsdf_index);
        const std::string frame_expr = layering.bsdfs[bsdf_index].preserve_tangent_frame
            ? fmt::format(
                  "Frame({}.bsdf_n[{}], {}.bsdf_t[{}], 1.0, Frame::Component::tangent)",
                  stack_name,
                  bsdf_index,
                  stack_name,
                  bsdf_index
              )
            : fmt::format("Frame({}.bsdf_n[{}], {}.bsdf_t[{}])", stack_name, bsdf_index, stack_name, bsdf_index);
        append_line(
            fmt::format(
                "let {} = {}({}.bsdf{}, {}.bsdf_weights[{}], {}.bsdf_transmission_scales[{}], {})",
                value_name,
                alias_reference(aliases, aliases.weighted_aliases.at(ref), use_local_aliases),
                stack_name,
                bsdf_index,
                stack_name,
                bsdf_index,
                stack_name,
                bsdf_index,
                frame_expr
            )
        );
        emitted_values[ref] = value_name;
        return value_name;
    }

    const std::size_t combiner_index = ref.combiner_index();
    const LayeringDesc::CombinerDesc& combiner = layering.combiners[combiner_index];
    const ClosureRef a_ref = combiner.children[0];
    const ClosureRef b_ref = combiner.children[1];
    const std::string a = build_closure_tree_composition_value_text(
        layering,
        aliases,
        a_ref,
        stack_name,
        use_local_aliases,
        line_prefix,
        emitted_values,
        out
    );
    const std::string b = build_closure_tree_composition_value_text(
        layering,
        aliases,
        b_ref,
        stack_name,
        use_local_aliases,
        line_prefix,
        emitted_values,
        out
    );
    const std::string core_name = fmt::format("{}_{}", combiner_value_prefix(combiner.mode), combiner_index);
    const std::string weighted_name = fmt::format("{}_weighted", core_name);

    switch (combiner.mode) {
    case CombinerMode::layer:
        append_line(
            fmt::format(
                "let {} = {}({}, {})",
                core_name,
                alias_reference(aliases, aliases.core_aliases.at(ref), use_local_aliases),
                a,
                b
            )
        );
        break;
    case CombinerMode::mix:
        append_line(
            fmt::format(
                "let {} = {}({}, {}, {}.mix_values[{}])",
                core_name,
                alias_reference(aliases, aliases.core_aliases.at(ref), use_local_aliases),
                a,
                b,
                stack_name,
                combiner_index
            )
        );
        break;
    case CombinerMode::add:
        append_line(
            fmt::format(
                "let {} = {}({}, {})",
                core_name,
                alias_reference(aliases, aliases.core_aliases.at(ref), use_local_aliases),
                a,
                b
            )
        );
        break;
    case CombinerMode::weighted_layer:
        FALCOR_UNREACHABLE();
    }

    append_line(
        fmt::format(
            "let {} = {}({}, {}.layer_weights[{}], {}.layer_transmission_scales[{}])",
            weighted_name,
            alias_reference(aliases, aliases.weighted_aliases.at(ref), use_local_aliases),
            core_name,
            stack_name,
            combiner_index,
            stack_name,
            combiner_index
        )
    );
    emitted_values[ref] = weighted_name;
    return weighted_name;
}

void emit_closure_tree_material_structs(
    const MaterialX::ShaderGenerator& shadergen,
    const ClosureTreeMaterialStructsDesc& params,
    MaterialX::ShaderStage& stage
)
{
    const std::string& material_name = params.material_name;
    const std::string& material_instance_name = params.material_instance_name;
    const std::string& material_data_name = params.material_data_name;
    const std::string& graph_name = params.graph_name;
    const LayeringDesc& layering = params.layering;
    const CodeGenDesc& desc = params.codegen_desc;
    const ProfileDesc& profile = params.profile;
    const bool graph_has_emission = params.graph_has_emission;
    std::unique_ptr<RootStrategy> strategy = make_root_strategy(layering, desc);
    const std::string aliases_name = profile.generated_prefix + "Aliases_$HASH";
    const std::string root_type_name = profile.generated_prefix + "FlatRootBSDF_$HASH";
    const CompositionAliases composition_aliases
        = strategy->build_aliases(layering, profile, aliases_name, root_type_name);
    const ClosureRef root = layering.main_layer;
    const std::string root_lobe_expr = root_lobes(layering, root);
    const std::string lobes
        = root.is_none() ? "LobeTypes::none" : (root_lobe_expr.empty() ? "LobeTypes::all" : root_lobe_expr);
    const bool graph_can_emit = graph_has_emission && graph_output_can_emit(stage.getOutputBlock(k_outputs));
    const std::string graph_emission = graph_emission_expr(stage.getOutputBlock(k_outputs));
    const std::string graph_opacity = graph_opacity_expr(stage.getOutputBlock(k_outputs));
    const std::string graph_thin_walled = graph_thin_walled_expr(stage.getOutputBlock(k_outputs));
    const bool use_synthetic_opacity_mix = synthetic_opacity_combiner_index(layering).has_value();

    shadergen.emitLine("#define MATERIAL_PAYLOAD_LIMIT " + std::to_string(desc.payload_size), stage, false);
    shadergen.emitLine("#define MATERIALX_PAYLOAD_SIZE " + std::to_string(params.materialx_payload_size), stage, false);
    shadergen.emitLineBreak(stage);
    strategy->emit_root_types(shadergen, layering, profile, root_type_name, stage);
    emit_closure_tree_composition_aliases(shadergen, composition_aliases, profile, stage);

    const std::string body_indent = "            ";
    const std::string stack_name = "stack";
    const std::string wi_expr = "result.shading_frame_ws.to_local(si.wi_ws)";

    const std::string stack_decl
        = use_synthetic_opacity_mix ? "var stack = mtlx_graph.mx_stack" : "let stack = mtlx_graph.mx_stack";

    const std::string transmissive_flag_line = root_transmissive(layering, root)
        ? body_indent + "result.flags = result.flags | MaterialProperties::Flags::is_transmissive;\n"
        : std::string{};
    const std::string curve_scattering_flag_line = root_curve_scattering(layering, root)
        ? body_indent + "result.flags = result.flags | MaterialProperties::Flags::is_curve_scattering;\n"
        : std::string{};

    const std::string synthetic_opacity_lines = build_synthetic_opacity_stack_setup_text(
        layering,
        stack_name,
        graph_opacity,
        strategy->stack_params().use_stack_frames,
        body_indent
    );

    const RootStrategy::RootValueBlock root_block = strategy->build_root_value_text(
        layering,
        profile,
        root_type_name,
        stack_name,
        wi_expr,
        composition_aliases,
        body_indent
    );
    const std::string collect_extra_method = strategy->build_collect_extra_text(layering, "    ");

    std::string source = stage.getSourceCode();
    source += codegen_support::render_snippet(
        "methods/basic/basic_material_wrapper.slang",
        codegen_support::TokenMap{
            {"$MxMaterialInstanceName", material_instance_name},
            {"$MxMaterialName", material_name},
            {"$MxMaterialDataName", material_data_name},
            {"$MxGraphName", graph_name},
            {"$MxAliasesName", composition_aliases.namespace_name},
            {"$MxLobes", lobes},
            {"$MxTransmissiveFlagLine", transmissive_flag_line},
            {"$MxCurveScatteringFlagLine", curve_scattering_flag_line},
            {"$MxGraphEmission", graph_emission},
            {"$MxGraphThinWalled", graph_thin_walled},
            {"$MxGraphCanEmit", graph_can_emit ? "true" : "false"},
            {"$MxStackDecl", stack_decl},
            {"$MxSyntheticOpacityLines", synthetic_opacity_lines},
            {"$MxRootValueLines", root_block.lines},
            {"$MxRootValueExpr", root_block.root_expr},
            {"$MxCollectExtraBsdfPropertiesMethod", collect_extra_method},
        }
    );
    stage.setSourceCode(source);
    shadergen.emitLineBreak(stage);
}

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
