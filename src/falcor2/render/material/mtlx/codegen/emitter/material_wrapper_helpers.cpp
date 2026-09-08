// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_wrapper_helpers.h"

#include <MaterialXCore/Types.h>

#include <algorithm>
#include <array>
#include <set>

namespace falcor {
namespace mtlx {
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
            "BSDFFlags::diffuse_reflection",
            "BSDFFlags::diffuse_transmission",
            "BSDFFlags::glossy_reflection",
            "BSDFFlags::glossy_transmission",
            "BSDFFlags::glossy_curve",
            "BSDFFlags::delta_transmission",
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

} // namespace emitter
} // namespace mtlx
} // namespace falcor
