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

constexpr std::size_t k_extra_bsdf_properties_max_bsdf_count = 16;

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
    append("    ctx.wi_adjusted = shading_frame_ws.to_local(si.wi_ws);");
    append("    ctx.closure_weight = float3(1.0);");
    out += report_lines;
    append("    Frame shading_from_adjusted_shading = {};");
    append("    shading_from_adjusted_shading.normal = si.shading_frame_ws.to_local(shading_frame_ws.normal);");
    append("    shading_from_adjusted_shading.tangent = si.shading_frame_ws.to_local(shading_frame_ws.tangent);");
    append("    shading_from_adjusted_shading.bitangent = si.shading_frame_ws.to_local(shading_frame_ws.bitangent);");
    append("    // Internal collectors write BSDF frames in adjusted-shading-frame space.");
    append("    // Keep this fixed bound so [ForceUnroll] flattens the reporting-space conversion.");
    append("    [ForceUnroll]");
    append("    for (int i = 0; i < ExtraBsdfProperties::max_bsdf_count; ++i)");
    append("    {");
    append("        if (i < reported_count)");
    append("        {");
    append("            result.bsdf_N[i] = shading_from_adjusted_shading.to_global(result.bsdf_N[i]);");
    append("            result.bsdf_T[i] = shading_from_adjusted_shading.to_global(result.bsdf_T[i]);");
    append("            result.bsdf_B[i] = shading_from_adjusted_shading.to_global(result.bsdf_B[i]);");
    append("        }");
    append("    }");
    append("    result.bsdf_count = uint(reported_count);");
    append("    return result;");
    append("}");
    return out;
}

std::string build_collect_properties_method_text(
    const LayeringDesc& layering,
    const std::string& line_prefix,
    bool is_transmissive,
    bool is_curve_scattering,
    const std::string& report_lines
)
{
    std::string out;
    auto append = [&](const std::string& line)
    {
        out += line_prefix;
        out += line;
        out += '\n';
    };

    append("public MaterialProperties collect_properties(const SurfaceInteraction si)");
    append("{");
    append("    float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);");
    append("    MaterialProperties result = {};");
    append("    result.emission = emissive_radiance;");
    append("    if (thin_walled)");
    append("        result.flags = result.flags | MaterialProperties::Flags::is_thin_walled;");
    if (is_transmissive)
        append("    result.flags = result.flags | MaterialProperties::Flags::is_transmissive;");
    if (is_curve_scattering)
        append("    result.flags = result.flags | MaterialProperties::Flags::is_curve_scattering;");

    if (layering.main_layer.is_none()) {
        append("    return result;");
        append("}");
        return out;
    }

    if (report_lines.empty()) {
        append("    AlbedoContributions albedo = bsdf.eval_albedo(wi_ls);");
        append("    RoughnessInformation roughness = bsdf.eval_roughness(wi_ls);");
        append("    result.roughness = roughness.roughness.x;");
        append("    result.guide_normal = shading_frame_ws.normal;");
        append("    result.specular_reflection_albedo = albedo.reflection;");
        if (is_transmissive)
            append("    result.specular_transmission_albedo = albedo.transmission;");
        append("    result.specular_reflectance = bsdf.ior_as_reflectance;");
        append("    return result;");
        append("}");
        return out;
    }

    append("");
    append("    float roughness_norm = 0.0;");
    append("    float3 guide_normal = float3(0.0);");
    append("    MxExtraBsdfPropertiesContext ctx = {};");
    append("    ctx.wi_adjusted = wi_ls;");
    append("    ctx.closure_weight = float3(1.0);");
    out += report_lines;
    append("");
    append("    AlbedoContributions albedo = bsdf.eval_albedo(wi_ls);");
    append("    RoughnessInformation root_roughness = bsdf.eval_roughness(wi_ls);");
    append(
        "    float root_roughness_guide = clamp((root_roughness.roughness.x + root_roughness.roughness.y) * 0.5, 0.0, "
        "1.0);"
    );
    append("    result.roughness = roughness_norm > 0.0 ? clamp(result.roughness / roughness_norm, 0.0, 1.0)");
    append("                                            : root_roughness_guide;");
    append(
        "    result.guide_normal = dot(guide_normal, guide_normal) > 0.0 ? "
        "normalize(shading_frame_ws.to_global(guide_normal)) : "
        "shading_frame_ws.normal;"
    );
    append("");
    if (is_transmissive) {
        append("    float transmission_split = min(root_roughness.roughness.x + root_roughness.roughness.y, 1.0);");
        append("    result.diffuse_transmission_albedo = albedo.transmission * transmission_split;");
        append("    result.specular_transmission_albedo = albedo.transmission * (1.0 - transmission_split);");
    }
    append("    result.specular_reflectance = result.specular_reflection_albedo;");
    append("    return result;");
    append("}");
    return out;
}

class ClosureTreeExtraLeafIndexBuilder {
public:
    explicit ClosureTreeExtraLeafIndexBuilder(const LayeringDesc& layering)
        : m_layering(layering)
    {
    }

    std::vector<std::size_t> build()
    {
        append_ref(m_layering.main_layer);
        return m_leaf_visit_to_bsdf_index;
    }

private:
    void append_ref(ClosureRef ref)
    {
        if (ref.is_none())
            return;

        if (ref.is_bsdf()) {
            m_leaf_visit_to_bsdf_index.push_back(ref.bsdf_index());
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
    std::vector<std::size_t> m_leaf_visit_to_bsdf_index;
};

std::string build_closure_tree_collect_extra_text(const LayeringDesc& layering, const std::string& line_prefix)
{
    if (layering.main_layer.is_none())
        return {};

    const std::vector<std::size_t> leaf_visit_to_bsdf_index = ClosureTreeExtraLeafIndexBuilder(layering).build();

    std::string body_lines;
    auto append = [&](const std::string& line)
    {
        body_lines += line_prefix;
        body_lines += "    ";
        body_lines += line;
        body_lines += ";\n";
    };

    append("ctx.leaf_visit_index = 0");
    for (std::size_t leaf_visit_index = 0; leaf_visit_index < leaf_visit_to_bsdf_index.size(); ++leaf_visit_index) {
        if (leaf_visit_index >= k_extra_bsdf_properties_max_bsdf_count)
            break;
        const std::size_t bsdf_index = leaf_visit_to_bsdf_index.size() > k_extra_bsdf_properties_max_bsdf_count
            ? leaf_visit_index
            : leaf_visit_to_bsdf_index[leaf_visit_index];
        append(
            "ctx.leaf_visit_to_bsdf_index[" + std::to_string(leaf_visit_index) + "] = " + std::to_string(bsdf_index)
        );
    }
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

    std::string build_collect_properties_text(
        const LayeringDesc& layering,
        const std::string& line_prefix,
        bool is_transmissive,
        bool is_curve_scattering
    ) const override
    {
        return build_collect_properties_method_text(
            layering,
            line_prefix,
            is_transmissive,
            is_curve_scattering,
            line_prefix + "    bsdf.collect_material_properties(result, roughness_norm, guide_normal, ctx);\n"
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

    std::string build_collect_properties_text(
        const LayeringDesc& layering,
        const std::string& line_prefix,
        bool is_transmissive,
        bool is_curve_scattering
    ) const override
    {
        return build_collect_properties_method_text(
            layering,
            line_prefix,
            is_transmissive,
            is_curve_scattering,
            line_prefix + "    bsdf.collect_material_properties(result, roughness_norm, guide_normal, ctx);\n"
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
