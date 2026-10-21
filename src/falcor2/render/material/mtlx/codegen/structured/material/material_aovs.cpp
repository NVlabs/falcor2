// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_aovs.h"

#include "falcor2/core/error.h"

#include <fmt/format.h>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

namespace {

std::string padded_value(std::string_view value, uint32_t component_count)
{
    switch (component_count) {
    case 1:
        return fmt::format("float4({}, 0.0, 0.0, 0.0)", value);
    case 2:
        return fmt::format("float4({}, 0.0, 0.0)", value);
    case 3:
        return fmt::format("float4({}, 0.0)", value);
    case 4:
        return fmt::format("float4({})", value);
    default:
        FALCOR_UNREACHABLE();
    }
}

} // namespace

std::string emit_material_aov_writer(
    const mx::VariableBlock& outputs,
    const CodeGenResult& result,
    std::string_view graph_name,
    std::string_view material_data_name
)
{
    if (result.material_aov_outputs.empty())
        return {};
    std::string source
        = R"(    public override void write_material_aovs_impl<Writer : IMaterialAOVWriter, LodSampler : ILodSampler>(
        uint2 coord, const SurfaceInteraction si, const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        if (!()";
    for (size_t i = 0; i < result.material_aov_outputs.size(); ++i) {
        if (i)
            source += " || ";
        source += fmt::format("Writer::generic_data_enabled({}u)", result.material_aov_outputs[i].id);
    }
    source += "))\n            return;\n";
    source += fmt::format(
        R"(#if MATERIALX_PAYLOAD_SIZE > MATERIAL_PAYLOAD_LIMIT
        let data = data_buffer.load<{}>(0);
#endif
        typealias Graph = {}<LodSampler>;
        SurfaceInteraction graph_si = prepare_graph_si(si, hints);
        let mtlx_graph = Graph(data, graph_si, lod_sampler, uint(hints));
)",
        material_data_name,
        graph_name
    );
    for (const AOVDesc& material_aov : result.material_aov_outputs) {
        const mx::ShaderPort* graph_output = nullptr;
        for (size_t i = 1; i < outputs.size(); ++i) {
            if (outputs[i]->getName() == material_aov.name) {
                graph_output = outputs[i];
                break;
            }
        }
        FALCOR_CHECK(
            graph_output,
            "MTLX selected material AOV '{}' is not present in the generated Graph outputs.",
            material_aov.name
        );
        const std::string value = "mtlx_graph." + graph_output->getVariable();
        source += fmt::format("        if (Writer::generic_data_enabled({0}u))\n", material_aov.id);
        source += fmt::format(
            "            Writer::generic_data(coord, {0}u, {1});\n",
            material_aov.id,
            padded_value(value, material_aov.component_count)
        );
    }
    source += "    }\n";
    return source;
}

} // namespace falcor::mtlx::structured
