// SPDX-License-Identifier: Apache-2.0

#include "bsdf_mix_emit.h"

#include "../../../policy.h"
#include "../../../bsdf_mix_layering.h"

#include "falcor2/core/error.h"

#include <fmt/format.h>

#include <vector>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

namespace {

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

MxFlatLayerPassThroughAlbedoMode bsdf_mix_layer_pass_through_albedo_mode(const LayeringDesc::BsdfDesc& bsdf)
{
    // BSDL transmissive microfacet albedo() returns unity while the physical
    // IBSDF split reports the actual reflection/transmission shares. For the
    // reflection-only case, BSDL albedo() is only the reflected share; the
    // pass-through sits in eval_albedo().transmission for 1.38 layering.
    if (is_scatter_mode_bsdf(bsdf.node_impl))
        return MxFlatLayerPassThroughAlbedoMode::transmissive_unity;
    if (bsdf.node_impl == "IM_sheen_bsdf_genslangpt")
        return MxFlatLayerPassThroughAlbedoMode::reflection;
    return MxFlatLayerPassThroughAlbedoMode::scattering;
}

MxFlatLayerPassThroughTransmissionMode bsdf_mix_layer_pass_through_transmission_mode(const LayeringDesc::BsdfDesc& bsdf)
{
    // bsdf_mix compatibility rebuilds the BSDL filter_o pass-through from the
    // physical IBSDF split. The modes below encode the few BSDF families where
    // BSDL layering deliberately differs from through-material transmission.
    if (is_scatter_mode_bsdf(bsdf.node_impl))
        return MxFlatLayerPassThroughTransmissionMode::physical_unless_transmissive;
    if (bsdf.node_impl == "IM_sheen_bsdf_genslangpt")
        return MxFlatLayerPassThroughTransmissionMode::physical;
    if (is_unity_layer_pass_through_transmission_bsdf(bsdf))
        return MxFlatLayerPassThroughTransmissionMode::one;
    return MxFlatLayerPassThroughTransmissionMode::physical;
}

MxFlatRootDesc
make_mx139_bsdf_mix_root_desc(const LayeringDesc& desc, const ProfileDesc& profile, const std::string& root_type_name)
{
    FALCOR_UNUSED(profile);
    MxFlatRootDesc flat_desc;
    flat_desc.type_name = root_type_name;
    flat_desc.root_ref = desc.main_layer;
    flat_desc.bsdfs.reserve(desc.bsdfs.size());
    for (size_t i = 0; i < desc.bsdfs.size(); ++i) {
        const LayeringDesc::BsdfDesc& bsdf = desc.bsdfs[i];
        flat_desc.bsdfs.push_back(
            {fmt::format("bsdf{}", i),
             bsdf.bsdf_type,
             bsdf_mix_layer_pass_through_albedo_mode(bsdf),
             bsdf_mix_layer_pass_through_transmission_mode(bsdf),
             is_scatter_mode_bsdf(bsdf.node_impl)}
        );
    }

    flat_desc.combiners.reserve(desc.combiners.size());
    for (size_t i = 0; i < desc.combiners.size(); ++i) {
        const LayeringDesc::CombinerDesc& combiner = desc.combiners[i];
        MxFlatCombinerDesc flat_combiner;
        flat_combiner.index = int(i);
        flat_combiner.mode = bsdf_mix_mode(combiner.mode);
        if (combiner.mode == CombinerMode::mix) {
            flat_combiner.branch0_ref = combiner.children[1];
            flat_combiner.branch1_ref = combiner.children[0];
        } else {
            flat_combiner.branch0_ref = combiner.children[0];
            flat_combiner.branch1_ref = combiner.children[1];
        }
        flat_desc.combiners.push_back(flat_combiner);
    }
    return flat_desc;
}

} // namespace

void emit_bsdf_mix_root_bsdf_type(
    const MaterialX::ShaderGenerator& shadergen,
    const LayeringDesc& layering,
    const ProfileDesc& profile,
    const std::string& bsdf_mix_root_type,
    MaterialX::ShaderStage& stage
)
{
    shadergen.emitString(
        emit_bsdf_mix_root_bsdf(make_mx139_bsdf_mix_root_desc(layering, profile, bsdf_mix_root_type)),
        stage
    );
}

std::string emit_bsdf_mix_composition_value(
    const MaterialX::ShaderGenerator& shadergen,
    const LayeringDesc& layering,
    const CompositionAliases& aliases,
    const std::string& stack_name,
    const std::string& wi_expr,
    MaterialX::ShaderStage& stage,
    bool use_local_aliases
)
{
    if (layering.main_layer.is_none())
        return alias_reference(aliases, "RootBSDF", use_local_aliases) + "()";

    std::vector<std::string> args;
    args.reserve(layering.bsdfs.size() + 6);
    for (size_t i = 0; i < layering.bsdfs.size(); ++i)
        args.push_back(fmt::format("{}.bsdf{}", stack_name, i));

    args.push_back(fmt::format("{}.bsdf_weights", stack_name));
    args.push_back(fmt::format("{}.bsdf_transmission_scales", stack_name));
    args.push_back(fmt::format("{}.bsdf_frames", stack_name));

    if (!layering.combiners.empty()) {
        args.push_back(fmt::format("{}.layer_weights", stack_name));
        args.push_back(fmt::format("{}.mix_values", stack_name));
    }
    args.push_back(wi_expr);

    const std::string value_name = "root_bsdf";
    shadergen.emitLine(
        fmt::format("let {} = {}(", value_name, alias_reference(aliases, "RootBSDF", use_local_aliases)),
        stage,
        false
    );
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string suffix = i + 1 < args.size() ? "," : "";
        shadergen.emitLine("    " + args[i] + suffix, stage, false);
    }
    shadergen.emitLine(")", stage);
    return value_name;
}

std::string build_bsdf_mix_composition_value_text(
    const LayeringDesc& layering,
    const CompositionAliases& aliases,
    const std::string& stack_name,
    const std::string& wi_expr,
    bool use_local_aliases,
    const std::string& line_prefix,
    std::string& out
)
{
    if (layering.main_layer.is_none())
        return alias_reference(aliases, "RootBSDF", use_local_aliases) + "()";

    std::vector<std::string> args;
    args.reserve(layering.bsdfs.size() + 6);
    for (size_t i = 0; i < layering.bsdfs.size(); ++i)
        args.push_back(fmt::format("{}.bsdf{}", stack_name, i));

    args.push_back(fmt::format("{}.bsdf_weights", stack_name));
    args.push_back(fmt::format("{}.bsdf_transmission_scales", stack_name));
    args.push_back(fmt::format("{}.bsdf_frames", stack_name));

    if (!layering.combiners.empty()) {
        args.push_back(fmt::format("{}.layer_weights", stack_name));
        args.push_back(fmt::format("{}.mix_values", stack_name));
    }
    args.push_back(wi_expr);

    const std::string value_name = "root_bsdf";
    out += line_prefix;
    out += fmt::format("let {} = {}(", value_name, alias_reference(aliases, "RootBSDF", use_local_aliases));
    out += '\n';
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string suffix = i + 1 < args.size() ? "," : "";
        out += line_prefix;
        out += "    ";
        out += args[i];
        out += suffix;
        out += '\n';
    }
    out += line_prefix;
    out += ");\n";
    return value_name;
}

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
