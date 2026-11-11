// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "microfacet_bsdf.h"

#include "../user_data.h"
#include "bsdf_emission.h"

#include "falcor2/core/error.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <fmt/format.h>

#include <memory>
#include <string>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

namespace {

class GenslangPtMicrofacetBsdfNode final : public mx::ShaderNodeImpl {
public:
    explicit GenslangPtMicrofacetBsdfNode(MxMicrofacetKind kind)
        : m_kind(kind)
    {
    }

    void emitFunctionDefinition(const mx::ShaderNode&, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        emit_bsdf_preparation_helper(context, stage);
    }

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        auto user_data = context.getUserData<EmissionUserData>(USER_DATA_NAME);
        FALCOR_CHECK(user_data, "Structured MaterialX microfacet node has no request.");
        emitOutputVariables(node, context, stage);
        const mx::ShaderOutput* output = node.getOutput();
        FALCOR_CHECK(output, "Structured MaterialX microfacet node has no output.");
        const mx::ShaderGenerator& target = context.getShaderGenerator();
        const std::string output_name = output->getVariable();
        const std::string normal = input(node, "normal", context);
        const std::string tangent = input(node, "tangent", context);
        const auto found = user_data->node_emission_facts().microfacet_selections.find(&node);
        FALCOR_CHECK(
            found != user_data->node_emission_facts().microfacet_selections.end(),
            "Structured MaterialX microfacet node '{}' has no resolved emission policy.",
            node.getName()
        );
        emit_bsdf_initialization(
            target,
            stage,
            output_name,
            input(node, "weight", context),
            constructor(node, context, found->second),
            normal,
            tangent
        );
    }

private:
    static std::string input(const mx::ShaderNode& node, const std::string& name, mx::GenContext& context)
    {
        const mx::ShaderInput* value = node.getInput(name);
        FALCOR_CHECK(value, "Structured MaterialX microfacet node '{}' lacks input '{}'.", node.getName(), name);
        return context.getShaderGenerator().getUpstreamResult(value, context);
    }

    static std::string
    reflection_tint(MxScatterSelection selection, const std::string& tint, const std::string& scatter)
    {
        switch (selection) {
        case MxScatterSelection::runtime:
            return "((" + scatter + ") == 1 ? float3(0.0) : " + tint + ")";
        case MxScatterSelection::static_r:
        case MxScatterSelection::static_rt:
            return tint;
        case MxScatterSelection::static_t:
            return "float3(0.0)";
        }
        FALCOR_UNREACHABLE();
    }

    static std::string
    transmission_tint(MxScatterSelection selection, const std::string& tint, const std::string& scatter)
    {
        switch (selection) {
        case MxScatterSelection::runtime:
            return "((" + scatter + ") == 0 ? float3(0.0) : " + tint + ")";
        case MxScatterSelection::static_t:
        case MxScatterSelection::static_rt:
            return tint;
        case MxScatterSelection::static_r:
            return "float3(0.0)";
        }
        FALCOR_UNREACHABLE();
    }

    std::string constructor(const mx::ShaderNode& node, mx::GenContext& context, MxMicrofacetSelection selection) const
    {
        FALCOR_CHECK(selection.kind == m_kind, "Structured MaterialX microfacet emitter received the wrong family.");
        const std::string type = microfacet_bsdf_type(selection);
        const std::string rough = input(node, "roughness", context);
        const std::string retroreflective = input(node, "retroreflective", context);
        const std::string thickness = input(node, "thinfilm_thickness", context);
        const std::string thinfilm_ior = input(node, "thinfilm_ior", context);
        if (m_kind == MxMicrofacetKind::dielectric) {
            const std::string tint = input(node, "tint", context);
            const std::string scatter = input(node, "scatter_mode", context);
            const std::string ior = input(node, "ior", context);
            const std::string reflection = reflection_tint(selection.scatter, tint, scatter);
            const std::string transmission = transmission_tint(selection.scatter, tint, scatter);
            if (selection.fresnel == MxFresnelSelection::airy)
                return fmt::format(
                    "{}({}, MxDielectricAiryFresnel(clamp_dielectric_ior({}), !si.front_facing, {}, {}), {}, {}, "
                    "float3(0.0), {}, !si.front_facing)",
                    type,
                    rough,
                    ior,
                    thickness,
                    thinfilm_ior,
                    reflection,
                    transmission,
                    retroreflective
                );
            return fmt::format(
                "{}({}, MxDielectricFresnel(clamp_dielectric_ior({}), !si.front_facing), {}, {}, float3(0.0), {}, "
                "!si.front_facing)",
                type,
                rough,
                ior,
                reflection,
                transmission,
                retroreflective
            );
        }
        if (m_kind == MxMicrofacetKind::conductor) {
            if (selection.fresnel != MxFresnelSelection::airy)
                return fmt::format(
                    "{}({}, MxConductorFresnel({}, {}), {})",
                    type,
                    rough,
                    input(node, "ior", context),
                    input(node, "extinction", context),
                    retroreflective
                );
            return fmt::format(
                "{}({}, MxConductorAiryFresnel({}, {}, {}, {}), {})",
                type,
                rough,
                input(node, "ior", context),
                input(node, "extinction", context),
                thickness,
                thinfilm_ior,
                retroreflective
            );
        }
        const std::string scatter = input(node, "scatter_mode", context);
        const std::string color0 = input(node, "color0", context);
        const std::string color82 = input(node, "color82", context);
        const std::string color90 = input(node, "color90", context);
        const std::string exponent = input(node, "exponent", context);
        const std::string reflection = reflection_tint(selection.scatter, "float3(1.0)", scatter);
        const std::string transmission = transmission_tint(selection.scatter, "float3(1.0)", scatter);
        if (selection.fresnel == MxFresnelSelection::airy)
            return fmt::format(
                "{}({}, make_schlick_airy_fresnel({}, {}, {}, {}, !si.front_facing, {}, {}), {}, {}, {}, "
                "!si.front_facing)",
                type,
                rough,
                color0,
                color82,
                color90,
                exponent,
                thickness,
                thinfilm_ior,
                reflection,
                transmission,
                retroreflective
            );
        if (selection.fresnel == MxFresnelSelection::color82)
            return fmt::format(
                "{}({}, make_schlick_color82_fresnel({}, {}, {}, {}, !si.front_facing), {}, {}, {}, "
                "!si.front_facing)",
                type,
                rough,
                color0,
                color82,
                color90,
                exponent,
                reflection,
                transmission,
                retroreflective
            );
        return fmt::format(
            "{}({}, make_schlick_fresnel({}, {}, {}, !si.front_facing), {}, {}, {}, !si.front_facing)",
            type,
            rough,
            color0,
            color90,
            exponent,
            reflection,
            transmission,
            retroreflective
        );
    }

    MxMicrofacetKind m_kind;
};

} // namespace

mx::ShaderNodeImplPtr create_microfacet_dielectric_node()
{
    return std::make_shared<GenslangPtMicrofacetBsdfNode>(MxMicrofacetKind::dielectric);
}

mx::ShaderNodeImplPtr create_microfacet_conductor_node()
{
    return std::make_shared<GenslangPtMicrofacetBsdfNode>(MxMicrofacetKind::conductor);
}

mx::ShaderNodeImplPtr create_microfacet_generalized_schlick_node()
{
    return std::make_shared<GenslangPtMicrofacetBsdfNode>(MxMicrofacetKind::generalized_schlick);
}

} // namespace falcor::mtlx::structured
