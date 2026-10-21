// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_wrapper.h"

#include "../../material/analysis.h"
#include "../../material/material_aovs.h"
#include "../../target/shader_generator.h"
#include "../../target/snippet_renderer.h"
#include "../policy.h"

namespace falcor::mtlx::structured {

void emit_basic_material_wrapper(EmissionContext& context, const BasicMaterialWrapperDesc& desc)
{
    // The public basic representations differ only in their root type and its
    // construction. Keep the surrounding material contract in one visible
    // waterfall so empty surfaces, closure trees, and flat BSDF mixes cannot
    // drift in payload limits, opacity, or emission handling.
    context.shader_generator.emitLine(
        "#define MATERIAL_PAYLOAD_LIMIT " + std::to_string(context.request.payload_size),
        context.stage,
        false
    );
    context.shader_generator.emitLine(
        "#define MATERIALX_PAYLOAD_SIZE " + std::to_string(context.result.data_struct_size),
        context.stage,
        false
    );
    context.shader_generator.emitLineBreak(context.stage);

    if (!desc.root_type_source.empty()) {
        context.shader_generator.emitString(desc.root_type_source, context.stage);
        context.shader_generator.emitLineBreak(context.stage);
    }
    context.shader_generator.emitLine("public struct " + desc.aliases_name, context.stage, false);
    context.shader_generator.emitScopeBegin(context.stage);
    if (!desc.aliases_comment.empty())
        context.shader_generator.emitLine(desc.aliases_comment, context.stage, false);
    for (const BasicTypeAlias& alias : desc.aliases)
        context.shader_generator.emitLine("typealias " + alias.name + " = " + alias.type, context.stage);
    context.shader_generator.emitScopeEnd(context.stage, true);
    context.shader_generator.emitLineBreak(context.stage);

    const MaterialX::VariableBlock& outputs = context.stage.getOutputBlock("o");
    const bool has_material_output = output_block_has_material_output(outputs);
    const bool has_emission = context.ancillary.has_emission && has_material_output;
    const std::string output = outputs.empty() ? std::string{} : outputs[0]->getVariable();
    const bool dynamic_opacity = context.result.opacity.kind == CodeGenResult::OpacityResult::Kind::dynamic;
    const bool opaque = context.result.opacity.kind == CodeGenResult::OpacityResult::Kind::opaque;
    const std::string static_opacity = context.result.opacity.kind == CodeGenResult::OpacityResult::Kind::constant
        ? std::to_string(context.result.opacity.constant)
        : "1.0";
    const std::string material_aov_writer
        = emit_material_aov_writer(outputs, context.result, context.names.graph_name, context.names.material_data_name);

    const std::string instance_declaration = context.result.is_hair
        ? "typealias " + context.names.material_instance_name + " = MXChiangHairMaterialInstance;\n"
        : render_snippet(
              "methods/basic/basic_material_instance.slang",
              {
                  {"$MxMaterialInstanceName", context.names.material_instance_name},
                  {"$MxAliasesName", desc.aliases_name},
                  {"$MxLobes", desc.lobe_flags},
                  {"$MxCollectPropertiesMethod", desc.collect_properties_method},
                  {"$MxCollectExtraBsdfPropertiesMethod", desc.collect_extra_bsdf_properties_method},
              }
          );
    std::string source = context.stage.getSourceCode();
    source += render_snippet(
        "methods/basic/basic_material_wrapper.slang",
        {
            {"$MxMaterialInstanceName", context.names.material_instance_name},
            {"$MxMaterialInstanceDeclaration", instance_declaration},
            {"$MxIsHair", context.result.is_hair ? "true" : "false"},
            {"$MxWriteMaterialAOVsMethod", material_aov_writer},
            {"$MxMaterialName", context.names.material_name},
            {"$MxMaterialDataName", context.names.material_data_name},
            {"$MxGraphName", context.names.graph_name},
            {"$MxHasEmission", has_emission ? "1" : "0"},
            {"$MxEmissionPredicate", context.ancillary.emission_predicate},
            {"$MxEvaluateOpacity", dynamic_opacity ? "1" : "0"},
            {"$MxOpacityIsOpaque", opaque ? "1" : "0"},
            {"$MxStaticOpacity", static_opacity},
            {"$MxGraphEmission", has_material_output ? "mtlx_graph." + output + ".edf.radiance" : "float3(0.0)"},
            {"$MxGraphOpacity", has_material_output ? "clamp(mtlx_graph." + output + ".opacity, 0.0, 1.0)" : "1.0"},
            {"$MxAliasesName", desc.aliases_name},
            {"$MxLobes", desc.lobe_flags},
            {"$MxRootValueLines", desc.root_value_lines},
            {"$MxRootValueExpr", desc.root_value},
            {"$MxCollectPropertiesMethod", desc.collect_properties_method},
            {"$MxCollectExtraBsdfPropertiesMethod", desc.collect_extra_bsdf_properties_method},
        }
    );
    context.stage.setSourceCode(source);
    context.shader_generator.emitLineBreak(context.stage);
}

} // namespace falcor::mtlx::structured
