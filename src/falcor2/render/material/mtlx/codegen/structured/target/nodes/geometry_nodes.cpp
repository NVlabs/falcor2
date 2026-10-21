// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry_nodes.h"

#include "../user_data.h"

#include "falcor2/core/error.h"
#include "falcor2/render/geometry/geometry_attribute_classification.h"

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Geom.h>
#include <MaterialXCore/Value.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Util.h>
#include <MaterialXGenHw/HwConstants.h>

#include <fmt/format.h>

#include <charconv>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace falcor::mtlx::structured {

namespace {

std::string node_category(const mx::Document& document, const mx::ShaderNode& node)
{
    const mx::ImplementationPtr implementation = document.getImplementation(node.getImplementation().getName());
    const mx::NodeDefPtr node_def = implementation ? implementation->getNodeDef() : nullptr;
    return node_def ? node_def->getNodeString() : std::string();
}

/// Read an immutable input value, using MaterialX HwImplementation's terminology for inputs whose changes require
/// recompilation. Accept literals and baked root sockets, not runtime connections; this is stricter than "uniform".
std::string get_immutable_input_value(
    const mx::ShaderNode& node,
    const mx::ShaderStage& stage,
    std::string_view name,
    std::string_view fallback
)
{
    const mx::ShaderInput* input = node.getInput(std::string(name));
    if (!input)
        return std::string(fallback);

    const mx::ShaderOutput* connection = input->getConnection();
    if (!connection)
        return input->getValue() ? input->getValue()->getValueString() : std::string(fallback);

    // MaterialX represents baked root inputs as graph-socket connections.
    // Accept those sockets only when Falcor did not put them in the runtime
    // material-data block. Other connections can vary per call site.
    const mx::ShaderNode* owner = connection->getNode();
    if (owner->isAGraph() && !owner->getParent()) {
        const mx::VariableBlock& runtime_inputs = stage.getUniformBlock(mx::HW::PUBLIC_UNIFORMS);
        bool is_runtime = false;
        for (const mx::ShaderPort* runtime_input : runtime_inputs.getVariableOrder())
            is_runtime |= runtime_input == connection;
        if (!is_runtime)
            return connection->getValueString();
    }

    FALCOR_THROW(
        "MaterialX immutable input '{}' on node '{}' must be a compile-time value; it is connected to '{}' "
        "with value '{}'.",
        name,
        node.getName(),
        owner->getName(),
        connection->getValueString()
    );
}

uint32_t parse_index(std::string_view value, std::string_view owner)
{
    uint32_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    FALCOR_CHECK(
        error == std::errc() && end == value.data() + value.size(),
        "MaterialX {} must be a non-negative integer, got '{}'.",
        owner,
        value
    );
    return result;
}

/// Read an immutable MaterialX geometry-set index and require a non-negative integer.
uint32_t get_immutable_index(
    const mx::ShaderNode& node,
    const mx::ShaderStage& stage,
    std::string_view name = "index",
    uint32_t fallback = 0
)
{
    const std::string value = get_immutable_input_value(node, stage, name, std::to_string(fallback));
    return parse_index(value, fmt::format("immutable input '{}' on node '{}'", name, node.getName()));
}

/// Read an immutable MaterialX coordinate-space name supported by Falcor's GeomPropSpace.
std::string get_immutable_space(
    const mx::ShaderNode& node,
    const mx::ShaderStage& stage,
    std::string_view name,
    std::string_view fallback
)
{
    const std::string value = get_immutable_input_value(node, stage, name, fallback);
    FALCOR_CHECK(
        value == "model" || value == "object" || value == "world",
        "MaterialX immutable input '{}' on node '{}' has unsupported space '{}'.",
        name,
        node.getName(),
        value
    );
    return value;
}

bool is_fixed_attribute(MeshAttributeSemantic semantic, uint32_t index)
{
    return classify_geometry_attribute(semantic, index).kind == GeometryAttributeClassificationKind::fixed;
}

uint32_t geompropdef_index(const mx::GeomPropDef& definition)
{
    return definition.hasIndex()
        ? parse_index(definition.getIndex(), fmt::format("geometry property '{}' index", definition.getName()))
        : 0;
}

mx::GeomPropDefPtr validated_geompropdef(
    const mx::Document& document,
    const mx::ShaderNode& node,
    std::string_view name,
    std::string_view output_type
)
{
    mx::GeomPropDefPtr definition = document.getGeomPropDef(std::string(name));
    if (!definition)
        return nullptr;
    FALCOR_CHECK(
        definition->getType() == output_type,
        "MaterialX geometry property '{}' has type '{}', but node '{}' reads '{}'.",
        name,
        definition->getType(),
        node.getName(),
        output_type
    );
    const bool is_uniform = definition->hasAttribute(mx::ValueElement::UNIFORM_ATTRIBUTE)
        && definition->getTypedAttribute<bool>(mx::ValueElement::UNIFORM_ATTRIBUTE);
    FALCOR_CHECK(
        !is_uniform,
        "MaterialX geometry property '{}' is uniform, but varying geompropvalue node '{}' cannot read uniform geometry "
        "properties.",
        name,
        node.getName()
    );
    FALCOR_CHECK(
        !definition->hasAttribute(mx::ValueElement::UNIT_ATTRIBUTE)
            && !definition->hasAttribute(mx::ValueElement::UNITTYPE_ATTRIBUTE),
        "MaterialX geometry property '{}' uses unit metadata, which Falcor per-vertex geometry attributes do not "
        "support.",
        name
    );
    return definition;
}

std::string output_default(const mx::ShaderNode& node, const mx::ShaderGenerator& generator)
{
    const mx::ShaderOutput* output = node.getOutput();
    FALCOR_CHECK(output, "Structured MaterialX geometry node '{}' has no output.", node.getName());
    const mx::Syntax& syntax = generator.getSyntax();
    return output->getValue() ? syntax.getValue(output) : syntax.getDefaultValue(output->getType());
}

std::optional<std::string> getter_method(std::string_view output_type)
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

std::string texcoord_expression(
    uint32_t index,
    std::string_view output_type,
    const std::string& fallback,
    bool& uses_geometry_properties
)
{
    if (!is_fixed_attribute(MeshAttributeSemantic::tex_coord, index))
        return fallback;
    if (index == 0 && output_type == "vector2")
        return "si.uv";

    std::string effective_fallback = fallback;
    if (index == 0) {
        if (output_type == "vector3")
            effective_fallback = "float3(si.uv, 0.0)";
        else if (output_type == "vector4")
            effective_fallback = "float4(si.uv, 0.0, 0.0)";
    }
    FALCOR_CHECK(
        output_type == "vector2" || output_type == "vector3" || output_type == "vector4",
        "MaterialX texcoord has unsupported output type '{}'.",
        output_type
    );
    uses_geometry_properties = true;
    return fmt::format("si.get_geomprop_accessor().get_additional_texcoord({}u, {})", index, effective_fallback);
}

std::string geomcolor_expression(
    uint32_t index,
    std::string_view output_type,
    const std::string& fallback,
    bool& uses_geometry_properties
)
{
    if (!is_fixed_attribute(MeshAttributeSemantic::color, index))
        return fallback;

    std::string storage_fallback;
    std::string suffix;
    if (output_type == "float") {
        storage_fallback = fmt::format("float4({}, 0.0, 0.0, 0.0)", fallback);
        suffix = ".x";
    } else if (output_type == "color3") {
        storage_fallback = fmt::format("float4({}, 0.0)", fallback);
        suffix = ".rgb";
    } else if (output_type == "color4") {
        storage_fallback = fallback;
    } else {
        FALCOR_THROW("MaterialX geomcolor has unsupported output type '{}'.", output_type);
    }
    uses_geometry_properties = true;
    return fmt::format("(si.get_geomprop_accessor().get_color({}u, {})){}", index, storage_fallback, suffix);
}

/// Emit the Falcor enum spelling for an already validated MaterialX space name.
std::string geomprop_space(std::string_view space)
{
    if (space == "model")
        return "GeomPropSpace::model";
    if (space == "object")
        return "GeomPropSpace::object";
    return "GeomPropSpace::world";
}

/// Emit a MaterialX intrinsic geometric property from Falcor's world-space interaction, converting if requested.
std::string intrinsic_expression(
    std::string_view geomprop,
    uint32_t index,
    std::string_view space,
    const std::string& fallback,
    bool& uses_geometry_properties
)
{
    if ((geomprop == "tangent" || geomprop == "bitangent") && index != 0)
        return fallback;
    std::string value;
    if (geomprop == "position")
        value = "si.position_ws";
    else if (geomprop == "normal")
        value = "si.shading_frame_ws.normal";
    else if (geomprop == "tangent")
        value = "si.shading_frame_ws.tangent";
    else if (geomprop == "bitangent")
        value = "si.shading_frame_ws.bitangent";
    else if (geomprop == "viewdirection")
        value = "mx_viewdirection_vector3(si)";
    else
        FALCOR_THROW("MaterialX uses unsupported intrinsic geometry property '{}'.", geomprop);

    if (space == "world")
        return value;
    uses_geometry_properties = true;
    const std::string target_space = geomprop_space(space);
    if (geomprop == "position") {
        return fmt::format(
            "transform_point(si.get_geomprop_accessor().get_space_transform({}, GeomPropSpace::world), {})",
            target_space,
            value
        );
    }
    if (geomprop == "normal") {
        // MaterialX intrinsic normals are unit length in the requested space, including under instance scaling.
        return fmt::format(
            "normalize(mul((float3x3)transpose(si.get_geomprop_accessor().get_space_transform(GeomPropSpace::world, "
            "{})), {}))",
            target_space,
            value
        );
    }
    const std::string transformed = fmt::format(
        "mul(si.get_geomprop_accessor().get_space_transform({}, GeomPropSpace::world), float4({}, 0.0)).xyz",
        target_space,
        value
    );
    // View direction is a vector, not a normal; restore unit length after nonuniform instance scaling.
    return geomprop == "viewdirection" ? "normalize(" + transformed + ")" : transformed;
}

std::string geompropvalue_expression(
    const mx::Document& document,
    const CodeGenDesc& request,
    const mx::ShaderNode& node,
    const mx::ShaderStage& stage,
    const std::string& fallback,
    bool& uses_geometry_properties
)
{
    const std::string name = get_immutable_input_value(node, stage, "geomprop", "");
    if (name.empty())
        return fallback;
    const mx::ShaderOutput* output = node.getOutput();
    FALCOR_CHECK(output, "Structured MaterialX geompropvalue node '{}' has no output.", node.getName());
    const std::string output_type = output->getType().getName();
    const mx::GeomPropDefPtr definition = validated_geompropdef(document, node, name, output_type);
    if (definition) {
        const std::string& geomprop = definition->getGeomProp();
        if (!geomprop.empty()) {
            if (geomprop == "texcoord")
                return texcoord_expression(
                    geompropdef_index(*definition),
                    output_type,
                    fallback,
                    uses_geometry_properties
                );
            if (geomprop == "geomcolor")
                return geomcolor_expression(
                    geompropdef_index(*definition),
                    output_type,
                    fallback,
                    uses_geometry_properties
                );
            if (geomprop == "position" || geomprop == "normal" || geomprop == "tangent" || geomprop == "bitangent"
                || geomprop == "viewdirection") {
                const uint32_t index = geompropdef_index(*definition);
                // Unlike the other intrinsics, MaterialX viewdirection defaults to world space.
                const std::string space = definition->hasSpace() ? definition->getSpace()
                    : geomprop == "viewdirection"                ? "world"
                                                                 : "object";
                FALCOR_CHECK(
                    space == "model" || space == "object" || space == "world",
                    "MaterialX geometry property '{}' has unsupported space '{}'.",
                    name,
                    space
                );
                return intrinsic_expression(geomprop, index, space, fallback, uses_geometry_properties);
            }
            FALCOR_THROW("MaterialX geometry property '{}' uses unsupported geomprop token '{}'.", name, geomprop);
        }
    }

    FALCOR_CHECK(
        request.reserve_geometry_attribute_id,
        "MaterialX blind geometry property '{}' requires a Scene geometry-attribute ID resolver.",
        name
    );
    const std::optional<std::string> method = getter_method(output_type);
    FALCOR_CHECK(method, "MaterialX geometry property '{}' has unsupported type '{}'.", name, output_type);
    const uint32_t id = request.reserve_geometry_attribute_id(name);
    uses_geometry_properties = true;
    return fmt::format("si.get_geomprop_accessor().{}({}u, {})", *method, id, fallback);
}

void emit_expression(
    const mx::ShaderNode& node,
    mx::GenContext& context,
    mx::ShaderStage& stage,
    const std::string& expression
)
{
    const mx::ShaderOutput* output = node.getOutput();
    FALCOR_CHECK(output, "Structured MaterialX geometry node missing output.");
    const mx::ShaderGenerator& generator = context.getShaderGenerator();
    generator.emitLineBegin(stage);
    generator.emitOutput(output, true, false, context, stage);
    generator.emitString(" = " + expression, stage);
    generator.emitLineEnd(stage);
}

} // namespace

mx::ShaderNodeImplPtr TexCoordNode::create()
{
    return std::make_shared<TexCoordNode>();
}

bool TexCoordNode::isEditable(const mx::ShaderInput& input) const
{
    return input.getName() != "index";
}

void TexCoordNode::emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const
{
    const auto user_data = context.getUserData<EmissionUserData>(USER_DATA_NAME);
    FALCOR_CHECK(user_data, "Structured MaterialX geometry node missing generation user data.");
    const mx::ShaderGenerator& generator = context.getShaderGenerator();
    emit_expression(
        node,
        context,
        stage,
        texcoord_expression(
            get_immutable_index(node, stage),
            node.getOutput()->getType().getName(),
            output_default(node, generator),
            user_data->emission_scratch().uses_geometry_properties
        )
    );
}

mx::ShaderNodeImplPtr GeomPropValueNode::create()
{
    return std::make_shared<GeomPropValueNode>();
}

bool GeomPropValueNode::isEditable(const mx::ShaderInput& input) const
{
    return input.getName() != "geomprop";
}

mx::ShaderNodeImplPtr IntrinsicGeomPropNode::create()
{
    return std::make_shared<IntrinsicGeomPropNode>();
}

bool IntrinsicGeomPropNode::isEditable(const mx::ShaderInput& input) const
{
    return input.getName() != "index" && input.getName() != "space";
}

void IntrinsicGeomPropNode::emitFunctionCall(
    const mx::ShaderNode& node,
    mx::GenContext& context,
    mx::ShaderStage& stage
) const
{
    const auto user_data = context.getUserData<EmissionUserData>(USER_DATA_NAME);
    FALCOR_CHECK(user_data, "Structured MaterialX intrinsic geometry node missing generation user data.");
    const std::string category = node_category(user_data->document(), node);
    std::string fallback = output_default(node, context.getShaderGenerator());
    if (category == "tangent")
        fallback = "float3(1.0, 0.0, 0.0)";
    else if (category == "bitangent")
        fallback = "float3(0.0, 0.0, 1.0)";
    emit_expression(
        node,
        context,
        stage,
        intrinsic_expression(
            category,
            get_immutable_index(node, stage),
            get_immutable_space(node, stage, "space", category == "viewdirection" ? "world" : "object"),
            fallback,
            user_data->emission_scratch().uses_geometry_properties
        )
    );
}

const GeomPropValueNode::ResolvedExpression& GeomPropValueNode::resolve(
    const mx::Document& document,
    const CodeGenDesc& request,
    const mx::ShaderNode& node,
    const mx::ShaderStage& stage
) const
{
    if (auto it = m_expressions.find(&node); it != m_expressions.end())
        return it->second;
    const mx::ShaderInput* default_input = node.getInput("default");
    FALCOR_CHECK(default_input, "Structured MaterialX geompropvalue node '{}' has no default input.", node.getName());
    // The binding is fixed, but the upstream fallback can change with emission
    // context (for example, when a height-to-normal node resamples its inputs).
    ResolvedExpression resolved;
    resolved.expression = geompropvalue_expression(
        document,
        request,
        node,
        stage,
        "$MxGeomPropDefault",
        resolved.uses_geometry_properties
    );
    return m_expressions.emplace(&node, std::move(resolved)).first->second;
}

void GeomPropValueNode::emitFunctionCall(
    const mx::ShaderNode& node,
    mx::GenContext& context,
    mx::ShaderStage& stage
) const
{
    const auto user_data = context.getUserData<EmissionUserData>(USER_DATA_NAME);
    FALCOR_CHECK(user_data, "Structured MaterialX geompropvalue node missing generation user data.");
    const auto& resolved = resolve(user_data->document(), user_data->request(), node, stage);
    user_data->emission_scratch().uses_geometry_properties |= resolved.uses_geometry_properties;
    std::string expression = resolved.expression;
    const std::string fallback = context.getShaderGenerator().getUpstreamResult(node.getInput("default"), context);
    mx::tokenSubstitution({{"$MxGeomPropDefault", fallback}}, expression);
    emit_expression(node, context, stage, expression);
}

mx::ShaderNodeImplPtr GeomPropValueUniformNode::create()
{
    return std::make_shared<GeomPropValueUniformNode>();
}

void GeomPropValueUniformNode::emitFunctionCall(const mx::ShaderNode& node, mx::GenContext&, mx::ShaderStage&) const
{
    FALCOR_THROW(
        "MaterialX geompropvalueuniform node '{}' is unsupported; Falcor geometry attributes are per-vertex values.",
        node.getName()
    );
}

mx::ShaderNodeImplPtr GeomColorNode::create()
{
    return std::make_shared<GeomColorNode>();
}

bool GeomColorNode::isEditable(const mx::ShaderInput& input) const
{
    return input.getName() != "index";
}

void GeomColorNode::emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const
{
    const auto user_data = context.getUserData<EmissionUserData>(USER_DATA_NAME);
    FALCOR_CHECK(user_data, "Structured MaterialX geometry node missing generation user data.");
    const mx::ShaderGenerator& generator = context.getShaderGenerator();
    emit_expression(
        node,
        context,
        stage,
        geomcolor_expression(
            get_immutable_index(node, stage),
            node.getOutput()->getType().getName(),
            output_default(node, generator),
            user_data->emission_scratch().uses_geometry_properties
        )
    );
}

} // namespace falcor::mtlx::structured
