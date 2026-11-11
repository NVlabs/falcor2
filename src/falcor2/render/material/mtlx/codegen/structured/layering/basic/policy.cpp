// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../policy.h"
#include "../policy_factories.h"

#include "material_wrapper.h"

#include "../bsdf_mix/flat_root.h"
#include "../../target/shader_generator.h"

#include "falcor2/core/error.h"

#include <fmt/format.h>

#include <algorithm>
#include <map>
#include <utility>

namespace falcor::mtlx::structured {

namespace {

struct ClosureTreePlan {
    struct Alias {
        std::string name;
        std::string type;
    };

    std::string namespace_name;
    std::vector<Alias> aliases;
    std::map<ClosureRef, std::string> core_aliases;
    std::map<ClosureRef, std::string> weighted_aliases;
};

std::string combiner_prefix(CombinerMode mode)
{
    switch (mode) {
    case CombinerMode::layer:
        return "Layer";
    case CombinerMode::mix:
        return "Mix";
    case CombinerMode::add:
        return "Add";
    case CombinerMode::weighted_layer:
        FALCOR_UNREACHABLE();
    }
    FALCOR_UNREACHABLE();
}

std::string combiner_value_prefix(CombinerMode mode)
{
    switch (mode) {
    case CombinerMode::layer:
        return "layer";
    case CombinerMode::mix:
        return "mix";
    case CombinerMode::add:
        return "add";
    case CombinerMode::weighted_layer:
        FALCOR_UNREACHABLE();
    }
    FALCOR_UNREACHABLE();
}

std::string combiner_type(CombinerMode mode)
{
    switch (mode) {
    case CombinerMode::layer:
        return "MxLayerBSDF";
    case CombinerMode::mix:
        return "MxMixBSDF";
    case CombinerMode::add:
        return "MxAddBSDF";
    case CombinerMode::weighted_layer:
        FALCOR_UNREACHABLE();
    }
    FALCOR_UNREACHABLE();
}

void build_aliases_recursive(const ClassifiedLayering& layering, ClosureRef ref, ClosureTreePlan& plan)
{
    if (plan.weighted_aliases.contains(ref))
        return;
    if (ref.is_bsdf()) {
        const std::size_t index = ref.bsdf_index();
        const std::string alias = fmt::format("Bsdf{}", index);
        plan.core_aliases[ref] = layering.bsdfs[index].bsdf_type;
        plan.weighted_aliases[ref] = alias;
        plan.aliases.push_back(
            {alias, fmt::format("MxFramedWeightedBSDF<{}, {}>", layering.bsdfs[index].bsdf_type, index)}
        );
        return;
    }

    const std::size_t index = ref.combiner_index();
    const auto& combiner = layering.combiners[index];
    build_aliases_recursive(layering, combiner.children[0], plan);
    build_aliases_recursive(layering, combiner.children[1], plan);
    const std::string core = fmt::format("{}{}", combiner_prefix(combiner.mode), index);
    const std::string weighted = "Weighted" + core;
    plan.core_aliases[ref] = core;
    plan.weighted_aliases[ref] = weighted;
    plan.aliases.push_back(
        {core,
         fmt::format(
             "{}<{}, {}>",
             combiner_type(combiner.mode),
             plan.weighted_aliases.at(combiner.children[0]),
             plan.weighted_aliases.at(combiner.children[1])
         )}
    );
    plan.aliases.push_back({weighted, "Weighted<" + core + ">"});
}

ClosureTreePlan build_aliases(const ClassifiedLayering& layering)
{
    ClosureTreePlan plan;
    plan.namespace_name = "MxAliases_$HASH";
    if (layering.main_layer.is_none()) {
        plan.aliases.push_back({"RootBSDF", "MxNullBSDF"});
    } else {
        build_aliases_recursive(layering, layering.main_layer, plan);
        plan.aliases.push_back({"RootBSDF", plan.weighted_aliases.at(layering.main_layer)});
    }
    return plan;
}

std::string append_composition_value(
    const ClassifiedLayering& layering,
    const ClosureTreePlan& aliases,
    ClosureRef ref,
    std::map<ClosureRef, std::string>& values,
    std::string& lines
)
{
    if (auto found = values.find(ref); found != values.end())
        return found->second;
    if (ref.is_none())
        return "mtlx_ns::RootBSDF()";
    if (ref.is_bsdf()) {
        const std::size_t index = ref.bsdf_index();
        const std::string name = fmt::format("bsdf_{}", index);
        const auto& bsdf = layering.bsdfs[index];
        const std::string frame = bsdf.preserve_tangent_frame
            ? fmt::format("Frame(stack.bsdf_n[{}], stack.bsdf_t[{}], 1.0, Frame::Component::tangent)", index, index)
            : fmt::format("Frame(stack.bsdf_n[{}], stack.bsdf_t[{}])", index, index);
        lines += fmt::format(
            "            let {} = mtlx_ns::{}(stack.bsdf{}, stack.bsdf_weights[{}], "
            "stack.bsdf_transmission_scales[{}], {}, {});\n",
            name,
            aliases.weighted_aliases.at(ref),
            index,
            index,
            index,
            bsdf.through_material_transmissive ? "1.0" : "0.0",
            frame
        );
        values[ref] = name;
        return name;
    }

    const std::size_t index = ref.combiner_index();
    const auto& combiner = layering.combiners[index];
    const std::string a = append_composition_value(layering, aliases, combiner.children[0], values, lines);
    const std::string b = append_composition_value(layering, aliases, combiner.children[1], values, lines);
    const std::string core = fmt::format("{}_{}", combiner_value_prefix(combiner.mode), index);
    if (combiner.mode == CombinerMode::mix) {
        lines += fmt::format(
            "            let {} = mtlx_ns::{}({}, {}, stack.mix_values[{}]);\n",
            core,
            aliases.core_aliases.at(ref),
            a,
            b,
            index
        );
    } else {
        lines += fmt::format("            let {} = mtlx_ns::{}({}, {});\n", core, aliases.core_aliases.at(ref), a, b);
    }
    const std::string weighted = core + "_weighted";
    lines += fmt::format(
        "            let {} = mtlx_ns::{}({}, stack.layer_weights[{}], "
        "stack.layer_transmission_scales[{}]);\n",
        weighted,
        aliases.weighted_aliases.at(ref),
        core,
        index,
        index
    );
    values[ref] = weighted;
    return weighted;
}

std::string root_lobes(const ClassifiedLayering& layering)
{
    return format_bsdf_lobe_flags(reachable_lobe_flags(layering, layering.main_layer));
}

bool is_scatter_mode_bsdf(const ClassifiedLayering::BsdfDesc& bsdf)
{
    return structured::is_scatter_mode_bsdf(bsdf.kind);
}

MxFlatLayerPassThroughAlbedoMode bsdf_mix_layer_pass_through_albedo_mode(const ClassifiedLayering::BsdfDesc& bsdf)
{
    if (is_scatter_mode_bsdf(bsdf))
        return MxFlatLayerPassThroughAlbedoMode::transmissive_unity;
    if (bsdf.kind == BsdfKind::sheen)
        return MxFlatLayerPassThroughAlbedoMode::reflection;
    return MxFlatLayerPassThroughAlbedoMode::scattering;
}

MxFlatLayerPassThroughTransmissionMode
bsdf_mix_layer_pass_through_transmission_mode(const ClassifiedLayering::BsdfDesc& bsdf)
{
    if (is_scatter_mode_bsdf(bsdf))
        return MxFlatLayerPassThroughTransmissionMode::physical_unless_transmissive;
    if (bsdf.kind == BsdfKind::sheen)
        return MxFlatLayerPassThroughTransmissionMode::physical;
    if (bsdf.kind == BsdfKind::surface_unlit_transparent)
        return MxFlatLayerPassThroughTransmissionMode::one;
    return MxFlatLayerPassThroughTransmissionMode::physical;
}

MxFlatCombinerMode bsdf_mix_mode(CombinerMode mode)
{
    switch (mode) {
    case CombinerMode::layer:
        return MxFlatCombinerMode::layer;
    case CombinerMode::mix:
        return MxFlatCombinerMode::mix;
    case CombinerMode::add:
        return MxFlatCombinerMode::add;
    case CombinerMode::weighted_layer:
        FALCOR_UNREACHABLE();
    }
    FALCOR_UNREACHABLE();
}

MxFlatRootDesc make_bsdf_mix_root_desc(const ClassifiedLayering& layering)
{
    MxFlatRootDesc result;
    result.type_name = "MxFlatRootBSDF_$HASH";
    result.root_ref = layering.main_layer;
    result.bsdfs.reserve(layering.bsdfs.size());
    for (std::size_t i = 0; i < layering.bsdfs.size(); ++i) {
        const auto& bsdf = layering.bsdfs[i];
        result.bsdfs.push_back(
            {fmt::format("bsdf{}", i),
             bsdf.bsdf_type,
             bsdf_mix_layer_pass_through_albedo_mode(bsdf),
             bsdf_mix_layer_pass_through_transmission_mode(bsdf),
             is_scatter_mode_bsdf(bsdf)}
        );
    }
    result.combiners.reserve(layering.combiners.size());
    for (std::size_t i = 0; i < layering.combiners.size(); ++i) {
        const auto& combiner = layering.combiners[i];
        MxFlatCombinerDesc flat;
        flat.index = static_cast<int>(i);
        flat.mode = bsdf_mix_mode(combiner.mode);
        if (combiner.mode == CombinerMode::mix) {
            flat.branch0_ref = combiner.children[1];
            flat.branch1_ref = combiner.children[0];
        } else {
            flat.branch0_ref = combiner.children[0];
            flat.branch1_ref = combiner.children[1];
        }
        result.combiners.push_back(flat);
    }
    return result;
}

std::string build_bsdf_mix_root_value(const ClassifiedLayering& layering)
{
    std::string output = "            let root_bsdf = mtlx_ns::RootBSDF(\n";
    std::vector<std::string> args;
    args.reserve(layering.bsdfs.size() + 6);
    for (std::size_t i = 0; i < layering.bsdfs.size(); ++i)
        args.push_back(fmt::format("stack.bsdf{}", i));
    args.push_back("stack.bsdf_weights");
    args.push_back("stack.bsdf_transmission_scales");
    args.push_back("stack.bsdf_frames");
    if (!layering.combiners.empty()) {
        args.push_back("stack.layer_weights");
        args.push_back("stack.mix_values");
    }
    args.push_back("result.shading_frame_ws.to_local(si.wi_ws)");
    for (std::size_t i = 0; i < args.size(); ++i)
        output += "                " + args[i] + (i + 1 < args.size() ? ",\n" : "\n");
    output += "            );\n";
    return output;
}

bool root_transmissive(const ClassifiedLayering& layering, ClosureRef ref)
{
    if (ref.is_none())
        return false;
    if (ref.is_bsdf())
        return layering.bsdfs[ref.bsdf_index()].transmissive;
    const auto& children = layering.combiners[ref.combiner_index()].children;
    return std::ranges::any_of(
        children,
        [&](ClosureRef child)
        {
            return root_transmissive(layering, child);
        }
    );
}

void append_method_line(std::string& output, const std::string& line)
{
    output += "    ";
    output += line;
    output += '\n';
}

std::string collect_extra_properties_method(const ClassifiedLayering& layering)
{
    if (layering.main_layer.is_none())
        return {};
    std::string output;
    append_method_line(output, "public override Optional<ExtraBsdfProperties>");
    append_method_line(output, "collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)");
    append_method_line(output, "{");
    append_method_line(output, "    ExtraBsdfProperties result = {};");
    append_method_line(output, "    int reported_count = 0;");
    append_method_line(output, "    MxExtraBsdfPropertiesContext ctx = {};");
    append_method_line(output, "    ctx.wi_adjusted = shading_frame_ws.to_local(si.wi_ws);");
    append_method_line(output, "    ctx.closure_weight = float3(1.0);");
    append_method_line(output, "    bsdf.collect_extra_bsdf_properties(result, reported_count, ctx);");
    append_method_line(output, "    Frame shading_from_adjusted_shading = {};");
    append_method_line(
        output,
        "    shading_from_adjusted_shading.normal = si.shading_frame_ws.to_local(shading_frame_ws.normal);"
    );
    append_method_line(
        output,
        "    shading_from_adjusted_shading.tangent = si.shading_frame_ws.to_local(shading_frame_ws.tangent);"
    );
    append_method_line(
        output,
        "    shading_from_adjusted_shading.bitangent = si.shading_frame_ws.to_local(shading_frame_ws.bitangent);"
    );
    append_method_line(output, "    // Internal collectors write BSDF frames in adjusted-shading-frame space.");
    append_method_line(
        output,
        "    // Keep this fixed bound so [ForceUnroll] flattens the reporting-space conversion."
    );
    append_method_line(output, "    [ForceUnroll]");
    append_method_line(output, "    for (int i = 0; i < ExtraBsdfProperties::max_bsdf_count; ++i)");
    append_method_line(output, "    {");
    append_method_line(output, "        if (i < reported_count)");
    append_method_line(output, "        {");
    append_method_line(
        output,
        "            result.bsdf_N[i] = shading_from_adjusted_shading.to_global(result.bsdf_N[i]);"
    );
    append_method_line(
        output,
        "            result.bsdf_T[i] = shading_from_adjusted_shading.to_global(result.bsdf_T[i]);"
    );
    append_method_line(
        output,
        "            result.bsdf_B[i] = shading_from_adjusted_shading.to_global(result.bsdf_B[i]);"
    );
    append_method_line(output, "        }");
    append_method_line(output, "    }");
    append_method_line(output, "    result.bsdf_count = uint(reported_count);");
    append_method_line(output, "    return result;");
    append_method_line(output, "}");
    return output;
}

std::string collect_properties_method(const ClassifiedLayering& layering)
{
    const bool empty = layering.main_layer.is_none();
    const bool transmissive = root_transmissive(layering, layering.main_layer);
    std::string output;
    append_method_line(output, "public MaterialProperties collect_properties(const SurfaceInteraction si)");
    append_method_line(output, "{");
    if (!empty)
        append_method_line(output, "    float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);");
    append_method_line(output, "    MaterialProperties result = {};");
    append_method_line(output, "    result.emission = emissive_radiance;");
    if (empty) {
        append_method_line(output, "    result.material_color = saturate(result.diffuse_reflection_albedo);");
        append_method_line(output, "    return result;");
        append_method_line(output, "}");
        return output;
    }
    append_method_line(output, "");
    append_method_line(output, "    float roughness_norm = 0.0;");
    append_method_line(output, "    float3 guide_normal = float3(0.0);");
    append_method_line(output, "    MxExtraBsdfPropertiesContext ctx = {};");
    append_method_line(output, "    ctx.wi_adjusted = wi_ls;");
    append_method_line(output, "    ctx.closure_weight = float3(1.0);");
    append_method_line(output, "    bsdf.collect_material_properties(result, roughness_norm, guide_normal, ctx);");
    append_method_line(output, "");
    append_method_line(output, "    AlbedoContributions albedo = bsdf.eval_albedo(wi_ls);");
    append_method_line(output, "    RoughnessInformation root_roughness = bsdf.eval_roughness(wi_ls);");
    append_method_line(
        output,
        "    float root_roughness_guide = clamp((root_roughness.roughness.x + root_roughness.roughness.y) * 0.5, 0.0, "
        "1.0);"
    );
    append_method_line(
        output,
        "    result.roughness = roughness_norm > 0.0 ? clamp(result.roughness / roughness_norm, 0.0, 1.0)"
    );
    append_method_line(output, "                                            : root_roughness_guide;");
    append_method_line(
        output,
        "    result.guide_normal = dot(guide_normal, guide_normal) > 0.0 ? "
        "normalize(shading_frame_ws.to_global(guide_normal)) : shading_frame_ws.normal;"
    );
    append_method_line(output, "");
    if (transmissive) {
        append_method_line(
            output,
            "    float transmission_split = min(root_roughness.roughness.x + root_roughness.roughness.y, 1.0);"
        );
        append_method_line(
            output,
            "    result.diffuse_transmission_albedo = albedo.transmission * transmission_split;"
        );
        append_method_line(
            output,
            "    result.specular_transmission_albedo = albedo.transmission * (1.0 - transmission_split);"
        );
    }
    append_method_line(output, "    result.specular_reflectance = result.specular_reflection_albedo;");
    append_method_line(output, "    result.material_color = saturate(result.diffuse_reflection_albedo);");
    append_method_line(output, "    return result;");
    append_method_line(output, "}");
    return output;
}

/// Emits the generic nested typed-BSDF representation used by the basic
/// material wrapper. Its generated aliases preserve the classified closure
/// tree directly.
class ClosureTreePolicy final : public ILayeringPolicy {
public:
    void analyze_original_layering(ClassifiedLayering&) override
    {
        FALCOR_CHECK(!m_built, "Structured closure_tree analysis ran after representation build.");
        // Closure-tree aliases are derived from the commonly lowered tree, so
        // this backend needs no facts from the original weighted layers.
    }

    void build_representation(ClassifiedLayering&& layering, const CodeGenDesc&) override
    {
        FALCOR_CHECK(
            layering.root_kind == SurfaceRootKind::non_empty,
            "Structured closure_tree received an empty surface."
        );
        FALCOR_CHECK(layering.weighted_layers_lowered, "Structured closure_tree received unlowered weighted layers.");
        // The classified tree is the representation. Build its stable aliases
        // now so every later query observes a complete backend state.
        m_layering = std::move(layering);
        m_aliases = build_aliases(m_layering);
        m_built = true;
    }

    GraphStoragePlan graph_storage_plan(const GeneratedNames&) const override
    {
        FALCOR_CHECK(m_built, "Structured closure_tree storage was queried before representation build.");
        return {};
    }

    void emit_representation_declarations(EmissionContext&) const override
    {
        FALCOR_CHECK(m_built, "Structured closure_tree declarations were emitted before representation build.");
        // The basic wrapper keeps its aliases with the post-graph material
        // implementation and therefore needs no declarations here.
    }

    void emit_material_implementation(EmissionContext& context) const override
    {
        FALCOR_CHECK(m_built, "Structured closure_tree material was emitted before representation build.");
        std::string root_lines;
        std::map<ClosureRef, std::string> values;
        const std::string root
            = append_composition_value(m_layering, m_aliases, m_layering.main_layer, values, root_lines);
        const std::string lobes = root_lobes(m_layering);
        BasicMaterialWrapperDesc desc;
        desc.aliases_name = m_aliases.namespace_name;
        desc.aliases_comment
            = "// Generated WeightedLayer<N> aliases below mean Weighted<Layer<N>> traversal wrappers.";
        desc.aliases.push_back({"Weighted<T : ILayerableBSDF>", "MxWeightedBSDF<T>"});
        for (const ClosureTreePlan::Alias& alias : m_aliases.aliases)
            desc.aliases.push_back({alias.name, alias.type});
        desc.lobe_flags = lobes.empty() ? "BSDFFlags::all" : lobes;
        desc.root_value_lines = std::move(root_lines);
        desc.root_value = root;
        desc.collect_properties_method = collect_properties_method(m_layering);
        desc.collect_extra_bsdf_properties_method = collect_extra_properties_method(m_layering);
        emit_basic_material_wrapper(context, desc);
    }

private:
    ClassifiedLayering m_layering;
    ClosureTreePlan m_aliases;
    bool m_built = false;
};

/// Emits the same basic material wrapper over a flattened BSDF-mix root. It
/// shares property and ancillary method generation with ClosureTreePolicy but
/// owns a different root representation and stack-frame requirement.
class BsdfMixPolicy final : public ILayeringPolicy {
public:
    void analyze_original_layering(ClassifiedLayering&) override
    {
        FALCOR_CHECK(!m_built, "Structured bsdf_mix analysis ran after representation build.");
        // Flattening operates on the commonly lowered tree, so this backend
        // needs no facts from the original weighted layers.
    }

    void build_representation(ClassifiedLayering&& layering, const CodeGenDesc&) override
    {
        FALCOR_CHECK(
            layering.root_kind == SurfaceRootKind::non_empty,
            "Structured bsdf_mix received an empty surface."
        );
        FALCOR_CHECK(layering.weighted_layers_lowered, "Structured bsdf_mix received unlowered weighted layers.");
        // Retain the lowered tree for property generation and derive the flat
        // root that the material wrapper will instantiate.
        m_layering = std::move(layering);
        m_root = make_bsdf_mix_root_desc(m_layering);
        m_built = true;
    }

    GraphStoragePlan graph_storage_plan(const GeneratedNames&) const override
    {
        FALCOR_CHECK(m_built, "Structured bsdf_mix storage was queried before representation build.");
        return GraphStoragePlan{.kind = GraphStorageKind::common_with_frames};
    }

    void emit_representation_declarations(EmissionContext&) const override
    {
        FALCOR_CHECK(m_built, "Structured bsdf_mix declarations were emitted before representation build.");
        // The flat root and aliases are emitted with the post-graph material
        // implementation, so no declaration is required before the graph.
    }

    void emit_material_implementation(EmissionContext& context) const override
    {
        FALCOR_CHECK(m_built, "Structured bsdf_mix material was emitted before representation build.");
        const std::string lobes = root_lobes(m_layering);
        emit_basic_material_wrapper(
            context,
            BasicMaterialWrapperDesc{
                .root_type_source = emit_bsdf_mix_root(m_root),
                .aliases_name = "MxAliases_$HASH",
                .aliases = {{"RootBSDF", "MxFlatRootBSDF_$HASH"}},
                .lobe_flags = lobes.empty() ? "BSDFFlags::all" : lobes,
                .root_value_lines = build_bsdf_mix_root_value(m_layering),
                .root_value = "root_bsdf",
                .collect_properties_method = collect_properties_method(m_layering),
                .collect_extra_bsdf_properties_method = collect_extra_properties_method(m_layering),
            }
        );
    }

private:
    ClassifiedLayering m_layering;
    MxFlatRootDesc m_root;
    bool m_built = false;
};

} // namespace

std::unique_ptr<ILayeringPolicy> create_closure_tree_policy()
{
    return std::make_unique<ClosureTreePolicy>();
}

std::unique_ptr<ILayeringPolicy> create_bsdf_mix_policy()
{
    return std::make_unique<BsdfMixPolicy>();
}

} // namespace falcor::mtlx::structured
