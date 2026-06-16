// SPDX-License-Identifier: Apache-2.0

#include "stack_data_basic_emit.h"

#include "../../../codegen_support/snippet_loader.h"

#include <fmt/format.h>

#include <algorithm>
#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

namespace mx = MaterialX;

void emit_basic_stack_data(
    const std::string& stack_data_type,
    const LayeringDesc& layering,
    bool use_stack_frames,
    mx::ShaderStage& stage
)
{
    const size_t bsdf_count = std::max(layering.bsdfs.size(), size_t(1));
    const size_t layer_count = std::max(layering.combiners.size(), size_t(1));

    std::string bsdf_frames_line;
    if (use_stack_frames)
        bsdf_frames_line = fmt::format("        Frame bsdf_frames[{}] = {{}};\n", bsdf_count);

    std::string bsdf_member_lines;
    for (size_t i = 0; i < layering.bsdfs.size(); ++i)
        bsdf_member_lines += fmt::format("        {} bsdf{} = {{}};\n", layering.bsdfs[i].bsdf_type, i);

    std::string set_bsdf_cases;
    if (layering.bsdfs.empty()) {
        set_bsdf_cases = "                default: break;\n";
    } else {
        for (size_t i = 0; i < layering.bsdfs.size(); ++i) {
            const std::string case_label = i + 1 < layering.bsdfs.size() ? fmt::format("case {}:", i) : "default:";
            set_bsdf_cases += fmt::format(
                "                {} bsdf{} = reinterpret<{}, TBSDF>(bsdf); break;\n",
                case_label,
                i,
                layering.bsdfs[i].bsdf_type
            );
        }
    }

    std::string init_vdf_bsdf_lines;
    for (size_t i = 0; i < bsdf_count; ++i)
        init_vdf_bsdf_lines += fmt::format("            bsdf_transmission_scales[{}] = float3(1.0);\n", i);
    std::string init_vdf_layer_lines;
    for (size_t i = 0; i < layer_count; ++i)
        init_vdf_layer_lines += fmt::format("            layer_transmission_scales[{}] = float3(1.0);\n", i);

    std::string source = stage.getSourceCode();
    source += codegen_support::render_snippet(
        "stack_data_basic.slang",
        codegen_support::TokenMap{
            {"$MxStackDataType", stack_data_type},
            {"$MxBsdfCount", std::to_string(bsdf_count)},
            {"$MxLayerCount", std::to_string(layer_count)},
            {"$MxBsdfFramesLine", bsdf_frames_line},
            {"$MxBsdfMemberLines", bsdf_member_lines},
            {"$MxSetBsdfCases", set_bsdf_cases},
            {"$MxInitVdfBsdfLines", init_vdf_bsdf_lines},
            {"$MxInitVdfLayerLines", init_vdf_layer_lines},
        }
    );
    stage.setSourceCode(source);
}

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
