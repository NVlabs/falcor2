// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "builtin_nodes.h"

#include "../codegen_support/shader_input_value.h"
#include "../codegen_support/string_utils.h"
#include "../../materialx_test_geomprop_ids.h"
#include "../codegen_user_data.h"

#include "falcor2/core/error.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <fmt/format.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace genslangpt {

using codegen_support::required_shader_input_value_string;

const char* const k_geomprop_input_name = "geomprop";

std::string shader_output_type_name(const mx::ShaderNode& node)
{
    const mx::ShaderOutput* output = node.getOutput();
    return output ? output->getType().getName() : "unknown";
}

namespace {

std::optional<std::string> convert_builtin_geomprop_expression(
    const std::string& base_expression,
    const std::string& base_type,
    const std::string& output_type
)
{
    if (output_type == base_type)
        return base_expression;

    if (base_type == "vector2" && (output_type == "vector3" || output_type == "color3"))
        return "float3(" + base_expression + ", 0.0)";
    if (base_type == "vector2" && (output_type == "vector4" || output_type == "color4"))
        return "float4(" + base_expression + ", 0.0, 0.0)";

    if (base_type == "vector3" && output_type == "vector2")
        return "(" + base_expression + ").xy";
    if (base_type == "vector3" && (output_type == "vector4" || output_type == "color4"))
        return "float4(" + base_expression + ", 0.0)";
    if (base_type == "vector3" && output_type == "color3")
        return base_expression;

    if ((base_type == "vector4" || base_type == "color4") && output_type == "float")
        return "(" + base_expression + ").x";
    if ((base_type == "vector4" || base_type == "color4") && output_type == "color3")
        return "(" + base_expression + ").rgb";
    if ((base_type == "vector4" || base_type == "color4") && output_type == "vector3")
        return "(" + base_expression + ").xyz";
    if (base_type == "vector4" && output_type == "color4")
        return base_expression;

    return {};
}

std::optional<std::string> builtin_geomprop_expression(const std::string& geomprop, const std::string& output_type)
{
    struct BuiltinGeomProp {
        std::string_view type;
        std::string_view expression;
    };
    static const std::unordered_map<std::string_view, BuiltinGeomProp> k_builtins = {
        {"UV0", {"vector2", "si.uv"}},
        {"UVMap", {"vector2", "si.uv"}},
        {"Pobject", {"vector3", "mul(object_from_world, float4(si.position_ws, 1.0)).xyz"}},
        {"Pworld", {"vector3", "si.position_ws"}},
        {"Nobject", {"vector3", "mul(object_from_world_it, si.shading_frame_ws.normal)"}},
        {"Nworld", {"vector3", "si.shading_frame_ws.normal"}},
        {"Tobject", {"vector3", "mul(object_from_world, float4(si.shading_frame_ws.tangent, 0.0)).xyz"}},
        {"Tworld", {"vector3", "si.shading_frame_ws.tangent"}},
        {"Bobject", {"vector3", "mul(object_from_world, float4(si.shading_frame_ws.bitangent, 0.0)).xyz"}},
        {"Bworld", {"vector3", "si.shading_frame_ws.bitangent"}},
        {"Vworld", {"vector3", "si.wi_ws"}},
    };
    auto it = k_builtins.find(geomprop);
    if (it == k_builtins.end())
        return {};
    return convert_builtin_geomprop_expression(
        std::string(it->second.expression),
        std::string(it->second.type),
        output_type
    );
}

std::optional<std::string> geomprop_provider_method(const std::string& output_type)
{
    if (output_type == "integer")
        return "get_int";
    if (output_type == "boolean")
        return "get_bool";
    if (output_type == "float")
        return "get_float";
    if (output_type == "vector2" || output_type == "color2")
        return "get_float2";
    if (output_type == "vector3" || output_type == "color3")
        return "get_float3";
    if (output_type == "vector4" || output_type == "color4")
        return "get_float4";
    return {};
}

std::string geomprop_fallback_expression(const std::string& output_type)
{
    if (output_type == "integer")
        return "0";
    if (output_type == "boolean")
        return "false";
    if (output_type == "float")
        return "0.0";
    if (output_type == "vector2" || output_type == "color2")
        return "si.uv";
    if (output_type == "vector3" || output_type == "color3")
        return "float3(si.uv, 0.0)";
    if (output_type == "vector4" || output_type == "color4")
        return "float4(si.uv, 0.0, 0.0)";
    return "0";
}

std::string
geomprop_provider_expression(const CodeGenDesc& desc, const std::string& geomprop, const std::string& output_type)
{
    if (auto builtin_expression = builtin_geomprop_expression(geomprop, output_type))
        return *builtin_expression;

    auto method = geomprop_provider_method(output_type);
    if (!method || geomprop.empty())
        return geomprop_fallback_expression(output_type);

    const std::optional<uint32_t> id = resolve_geomprop_id(desc, geomprop, output_type);
    if (!id)
        return geomprop_fallback_expression(output_type);

    return fmt::format("GeomPropProvider::{}(si, {})", *method, geomprop_id_const_name(geomprop));
}

std::string
texcoord_provider_expression(const CodeGenDesc& desc, const std::string& index, const std::string& output_type)
{
    if (index == "0")
        return output_type == "vector3" ? "float3(si.uv, 0.0)" : "si.uv";

    const std::string stream_name = "texcoord_" + index;
    const std::optional<uint32_t> id = resolve_geomprop_id(desc, stream_name, output_type);
    if (!id)
        return output_type == "vector3" ? "float3(si.uv, 0.0)" : "si.uv";

    const std::string const_name = geomprop_id_const_name(stream_name);
    const std::string provider_call = fmt::format("GeomPropProvider::get_float2(si, {})", const_name);
    return output_type == "vector3" ? fmt::format("float3({}, 0.0)", provider_call) : provider_call;
}

std::string geomcolor_fallback_expression(const std::string& output_type)
{
    if (output_type == "float")
        return "0.0";
    if (output_type == "color3")
        return "float3(0.0, 0.0, 0.0)";
    if (output_type == "color4")
        return "float4(0.0, 0.0, 0.0, 1.0)";
    return geomprop_fallback_expression(output_type);
}

std::string
geomcolor_provider_expression(const CodeGenDesc& desc, const std::string& index, const std::string& output_type)
{
    const std::string stream_name = "color_" + index;
    const std::optional<uint32_t> id = resolve_geomprop_id(desc, stream_name, "color4");
    if (!id)
        return geomcolor_fallback_expression(output_type);

    const std::string provider_call
        = fmt::format("GeomPropProvider::get_float4(si, {})", geomprop_id_const_name(stream_name));
    std::optional<std::string> expression = convert_builtin_geomprop_expression(provider_call, "color4", output_type);
    return expression ? *expression : geomcolor_fallback_expression(output_type);
}

} // namespace

std::string geomprop_id_const_name(const std::string& stream_name)
{
    return "mx_geomprop_id_" + codegen_support::sanitize_identifier(stream_name);
}

bool is_builtin_geomprop(const std::string& geomprop, const std::string& output_type)
{
    return builtin_geomprop_expression(geomprop, output_type).has_value();
}

std::optional<uint32_t>
resolve_geomprop_id(const CodeGenDesc& desc, std::string_view geomprop, std::string_view output_type)
{
    if (desc.geomprop_id_callback) {
        if (std::optional<uint32_t> id = desc.geomprop_id_callback(geomprop, output_type))
            return id;
    }
    return ::falcor::materialx::mx139::materialx_test_geomprop_id(geomprop);
}

mx::ShaderNodeImplPtr GeomPropValueNode::create()
{
    return std::make_shared<GeomPropValueNode>();
}

void GeomPropValueNode::emitFunctionCall(
    const mx::ShaderNode& node,
    mx::GenContext& context,
    mx::ShaderStage& stage
) const
{
    auto user_data = context.getUserData<CodegenUserData>("mx139");
    FALCOR_CHECK(user_data && user_data->inputs.desc, "MX139 geompropvalue node missing user data.");

    const std::string geomprop = required_shader_input_value_string(node, k_geomprop_input_name, "");
    const std::string output_type = shader_output_type_name(node);
    const mx::ShaderOutput* output = node.getOutput();
    FALCOR_CHECK(output, "MX139 geompropvalue node missing output.");
    const std::string expression = geomprop_provider_expression(*user_data->inputs.desc, geomprop, output_type);

    const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
    shadergen.emitLineBegin(stage);
    shadergen.emitOutput(output, true, false, context, stage);
    shadergen.emitString(" = " + expression, stage);
    shadergen.emitLineEnd(stage);
}

mx::ShaderNodeImplPtr TexCoordNode::create()
{
    return std::make_shared<TexCoordNode>();
}

void TexCoordNode::emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const
{
    auto user_data = context.getUserData<CodegenUserData>("mx139");
    FALCOR_CHECK(user_data && user_data->inputs.desc, "MX139 texcoord node missing user data.");

    const std::string index = required_shader_input_value_string(node, "index", "0");
    const std::string output_type = shader_output_type_name(node);
    const mx::ShaderOutput* output = node.getOutput();
    FALCOR_CHECK(output, "MX139 texcoord node missing output.");
    const std::string expression = texcoord_provider_expression(*user_data->inputs.desc, index, output_type);

    const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
    shadergen.emitLineBegin(stage);
    shadergen.emitOutput(output, true, false, context, stage);
    shadergen.emitString(" = " + expression, stage);
    shadergen.emitLineEnd(stage);
}

mx::ShaderNodeImplPtr GeomColorNode::create()
{
    return std::make_shared<GeomColorNode>();
}

void GeomColorNode::emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const
{
    auto user_data = context.getUserData<CodegenUserData>("mx139");
    FALCOR_CHECK(user_data && user_data->inputs.desc, "MX139 geomcolor node missing user data.");

    const std::string index = required_shader_input_value_string(node, "index", "0");
    const std::string output_type = shader_output_type_name(node);
    const mx::ShaderOutput* output = node.getOutput();
    FALCOR_CHECK(output, "MX139 geomcolor node missing output.");
    const std::string expression = geomcolor_provider_expression(*user_data->inputs.desc, index, output_type);

    const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
    shadergen.emitLineBegin(stage);
    shadergen.emitOutput(output, true, false, context, stage);
    shadergen.emitString(" = " + expression, stage);
    shadergen.emitLineEnd(stage);
}

} // namespace genslangpt
} // namespace mx139
} // namespace materialx
} // namespace falcor
