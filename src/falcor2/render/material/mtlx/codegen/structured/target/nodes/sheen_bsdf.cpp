// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sheen_bsdf.h"

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

class GenslangPtSheenBsdfNode final : public mx::ShaderNodeImpl {
public:
    void emitFunctionDefinition(const mx::ShaderNode&, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        emit_bsdf_preparation_helper(context, stage);
    }

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        auto user_data = context.getUserData<EmissionUserData>(USER_DATA_NAME);
        FALCOR_CHECK(user_data, "Structured MaterialX sheen node has no generation facts.");
        const auto found = user_data->node_emission_facts().sheen_selections.find(&node);
        FALCOR_CHECK(
            found != user_data->node_emission_facts().sheen_selections.end(),
            "Structured MaterialX sheen node '{}' has no resolved emission policy.",
            node.getName()
        );

        emitOutputVariables(node, context, stage);
        const mx::ShaderOutput* output = node.getOutput();
        FALCOR_CHECK(output, "Structured MaterialX sheen node has no output.");
        const mx::ShaderGenerator& target = context.getShaderGenerator();
        const std::string normal = input(node, "normal", context);
        const std::string bsdf_expression = fmt::format(
            "{}({}, {}, {}, !si.front_facing)",
            sheen_bsdf_type(found->second),
            input(node, "color", context),
            input(node, "roughness", context),
            input(node, "mode", context)
        );
        emit_bsdf_initialization(
            target,
            stage,
            output->getVariable(),
            input(node, "weight", context),
            bsdf_expression,
            normal,
            "si.shading_frame_ws.tangent"
        );
    }

private:
    static std::string input(const mx::ShaderNode& node, const std::string& name, mx::GenContext& context)
    {
        const mx::ShaderInput* value = node.getInput(name);
        FALCOR_CHECK(value, "Structured MaterialX sheen node '{}' lacks input '{}'.", node.getName(), name);
        return context.getShaderGenerator().getUpstreamResult(value, context);
    }
};

} // namespace

mx::ShaderNodeImplPtr create_sheen_bsdf_node()
{
    return std::make_shared<GenslangPtSheenBsdfNode>();
}

} // namespace falcor::mtlx::structured
