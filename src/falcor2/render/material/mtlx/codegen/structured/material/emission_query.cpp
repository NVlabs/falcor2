// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "emission_query.h"

#include "../graph/static_input_query/static_input_query.h"
#include "../target/nodes/source_code_node.h"
#include "../target/shader_generator.h"

#include <MaterialXGenShader/Nodes/MaterialNode.h>
#include <MaterialXGenShader/ShaderGenerator.h>

#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace falcor::mtlx::structured {
namespace mx = MaterialX;
namespace {

// A value can be unavailable without losing a useful necessary condition for
// nonzero emission, e.g. texture(uv) * current_uniform. False must prove zero;
// true includes unknown and is never a proof of positive emitted radiance.
struct Value {
    // Generated source for a value evaluable from constants and current material uniforms.
    // Missing source means evaluation needs shading state or an unsupported operation.
    std::optional<std::string> expression;
    // Generated boolean source, ready to emit in the query. The exact literals "false"
    // and "true" also encode proven zero and no useful restriction, respectively.
    std::string nonzero = "true";
};

// These helpers simplify source produced by this builder, with canonical boolean
// literals and side-effect-free predicates in the same query scope. Text equality
// is sufficient for idempotence ((p && p) == p, (p || p) == p). Different spellings of
// equivalent predicates merely miss a simplification; they do not change correctness.
std::string conjunction(const std::string& a, const std::string& b)
{
    if (a == "false" || b == "false")
        return "false";
    if (a == "true" || a == b)
        return b;
    if (b == "true")
        return a;
    return "(" + a + " && " + b + ")";
}

std::string disjunction(const std::string& a, const std::string& b)
{
    if (a == "true" || b == "true")
        return "true";
    if (a == "false" || a == b)
        return b;
    if (b == "false")
        return a;
    return "(" + a + " || " + b + ")";
}

Value expression_value(std::string expression)
{
    return {expression, "any((" + expression + ") != 0.0)"};
}

Value zero_value(std::string expression = "0.0")
{
    return {std::move(expression), "false"};
}

Value product(const Value& a, const Value& b)
{
    if (a.nonzero == "false" || b.nonzero == "false")
        return zero_value();
    if (a.expression && b.expression)
        return expression_value("(" + *a.expression + " * " + *b.expression + ")");
    return {{}, conjunction(a.nonzero, b.nonzero)};
}

Value mixed_value(const Value& foreground, const Value& background, const Value& weight, bool clamp_weight)
{
    if (!weight.expression)
        return {{}, disjunction(foreground.nonzero, background.nonzero)};
    const std::string w = clamp_weight ? "clamp(" + *weight.expression + ", 0.0, 1.0)" : *weight.expression;
    return {
        {},
        disjunction(
            conjunction("any((" + w + ") != 0.0)", foreground.nonzero),
            conjunction("any((" + w + ") != 1.0)", background.nonzero)
        )
    };
}

bool numeric_type(mx::TypeDesc type)
{
    return type == mx::Type::FLOAT || type == mx::Type::COLOR3 || type == mx::Type::COLOR4 || type == mx::Type::VECTOR2
        || type == mx::Type::VECTOR3 || type == mx::Type::VECTOR4 || type == mx::Type::INTEGER
        || type == mx::Type::BOOLEAN;
}

bool state_independent_operation(FormalNodeOperation operation)
{
    switch (operation) {
    case FormalNodeOperation::constant:
    case FormalNodeOperation::convert:
    case FormalNodeOperation::multiply:
    case FormalNodeOperation::add:
    case FormalNodeOperation::subtract:
    case FormalNodeOperation::divide:
    case FormalNodeOperation::mix:
    case FormalNodeOperation::clamp:
    case FormalNodeOperation::minimum:
    case FormalNodeOperation::maximum:
    case FormalNodeOperation::absolute_value:
    case FormalNodeOperation::invert:
    case FormalNodeOperation::luminance:
    case FormalNodeOperation::extract:
        return true;
    default:
        return false;
    }
}

// Each compound invocation has its own environment. A shared implementation
// graph must not substitute another caller's uniforms or merged default values.
struct Invocation {
    const mx::ShaderNode* caller = nullptr;
    Invocation* parent = nullptr;
    std::map<const mx::ShaderOutput*, Value> values;
    std::map<const mx::ShaderNode*, std::unique_ptr<Invocation>> children;
};

class EmissionQueryBuilder {
public:
    EmissionQueryBuilder(
        const mx::VariableBlock& uniforms,
        const StaticInputQuery& static_input_query,
        const GenslangPtShaderGenerator& generator,
        mx::GenContext& context
    )
        : m_uniforms(uniforms)
        , m_static_input_query(static_input_query)
        , m_generator(generator)
        , m_context(context)
        , m_calls(mx::Stage::PIXEL, generator.syntax_ptr())
        , m_definitions(mx::Stage::PIXEL, generator.syntax_ptr())
    {
    }

    EmissionQuery build(const mx::ShaderGraph& graph)
    {
        if (!graph.numOutputSockets())
            return {};
        Invocation root;
        const Value result = input_value(graph.getOutputSocket(0), root);
        if (result.nonzero == "true" || result.nonzero == "false")
            return {result.nonzero, {}};
        return {
            "mx_emission_is_nonzero(data)",
            m_definitions.getSourceCode() + "\nbool mx_emission_is_nonzero($MxMaterialDataName data)\n{\n"
                + m_calls.getSourceCode() + "return " + result.nonzero + ";\n}\n",
        };
    }

private:
    Value input_value(const mx::ShaderInput* input, Invocation& invocation)
    {
        if (!input)
            return {};
        if (numeric_type(input->getType())) {
            const auto value = m_static_input_query.input_value(*input);
            if (value.kind == ScalarStaticValueKind::KnownZero)
                return zero_value(m_generator.getSyntax().getDefaultValue(input->getType()));
        }
        if (const mx::ShaderOutput* output = input->getConnection())
            return output_value(*output, invocation);
        if (input->getType() == mx::Type::EDF)
            return zero_value();
        if (!numeric_type(input->getType()) || input->isBindInput() || !input->getValue())
            return {};
        return expression_value(m_generator.getSyntax().getValue(input->getType(), *input->getValue()));
    }

    Value output_value(const mx::ShaderOutput& output, Invocation& invocation)
    {
        const auto [it, inserted] = invocation.values.try_emplace(&output);
        if (!inserted)
            return it->second;
        // The unknown placeholder also makes a malformed recursive dependency conservative.
        it->second = resolve_output(output, invocation);
        // Keep shared closure paths linear in generated source as well as analysis work.
        // Numeric node values already have locals emitted by their implementations.
        if (it->second.nonzero != "true" && it->second.nonzero != "false"
            && (output.getType() == mx::Type::EDF || output.getType() == mx::Type::SURFACESHADER)) {
            const std::string name = "mx_emission_test_" + std::to_string(m_value_count++);
            if (it->second.expression) {
                m_generator.emitLine("float3 " + name + " = float3(" + *it->second.expression + ")", m_calls);
                it->second = expression_value(name);
            } else {
                m_generator.emitLine("bool " + name + " = " + it->second.nonzero, m_calls);
                it->second.nonzero = name;
            }
        }
        return it->second;
    }

    Value resolve_output(const mx::ShaderOutput& output, Invocation& invocation)
    {
        const mx::ShaderNode* node = output.getNode();
        if (!node)
            return {};
        if (node->isAGraph()) {
            if (invocation.caller)
                return input_value(invocation.caller->getInput(output.getName()), *invocation.parent);
            for (size_t i = 0; i < m_uniforms.size(); ++i) {
                const mx::ShaderPort* uniform = m_uniforms[i];
                if (uniform->getVariable() == output.getVariable() && numeric_type(output.getType()))
                    return expression_value("data." + uniform->getVariable());
            }
            if (numeric_type(output.getType()) && output.getValue())
                return expression_value(m_generator.getSyntax().getValue(output.getType(), *output.getValue()));
            return {};
        }
        if (const mx::ShaderGraph* inner = node->getImplementation().getGraph()) {
            auto& child = invocation.children[node];
            if (!child)
                child = std::make_unique<Invocation>(Invocation{node, &invocation, {}, {}});
            for (size_t i = 0; i < node->numOutputs() && i < inner->numOutputSockets(); ++i) {
                if (node->getOutput(i) == &output)
                    return input_value(inner->getOutputSocket(i), *child);
            }
            return {};
        }
        const auto input = [&](const char* name)
        {
            return input_value(node->getInput(name), invocation);
        };
        if (dynamic_cast<const mx::MaterialNode*>(&node->getImplementation()))
            return input(mx::ShaderNode::SURFACESHADER.c_str());

        const FormalNodeOperation operation = formal_node_operation(*node);
        // MaterialX dot is a passthrough. Its string "note" input is editor metadata,
        // not a value dependency; preserve both the input expression and its zero condition.
        if (operation == FormalNodeOperation::dot)
            return input("in");
        if (numeric_type(output.getType())) {
            Value result = numeric_value(*node, output, invocation);
            // Like MDL, a spatial product may still have an independently evaluable zero factor.
            // For texture(uv) * uniform, the texture contributes "true" (unknown), so
            // conjunction retains only the uniform's nonzero test without sampling the texture.
            if (!result.expression && operation == FormalNodeOperation::multiply)
                result.nonzero = conjunction(input("in1").nonzero, input("in2").nonzero);
            else if (!result.expression && operation == FormalNodeOperation::mix)
                result = mixed_value(input("fg"), input("bg"), input("mix"), false);
            return result;
        }
        if (output.getType() != mx::Type::EDF && output.getType() != mx::Type::SURFACESHADER)
            return {};
        switch (operation) {
        case FormalNodeOperation::uniform_edf:
            return input("color");
        case FormalNodeOperation::generalized_schlick_edf:
            // The direction-dependent coating multiplies its base; a zero base remains zero.
            return {{}, input("base").nonzero};
        case FormalNodeOperation::surface:
            return input("edf");
        case FormalNodeOperation::multiply:
            return product(input("in1"), input("in2"));
        case FormalNodeOperation::add: {
            const Value a = input("in1"), b = input("in2");
            if (a.expression && b.expression)
                return expression_value("(" + *a.expression + " + " + *b.expression + ")");
            return {{}, disjunction(a.nonzero, b.nonzero)};
        }
        case FormalNodeOperation::mix:
            return mixed_value(input("fg"), input("bg"), input("mix"), true);
        default:
            return {};
        }
    }

    Value numeric_value(const mx::ShaderNode& node, const mx::ShaderOutput& output, Invocation& invocation)
    {
        if (!state_independent_operation(formal_node_operation(node)) || node.numOutputs() != 1)
            return {};
        std::vector<Value> inputs;
        for (const mx::ShaderInput* input : node.getInputs()) {
            inputs.push_back(input_value(input, invocation));
            if (!inputs.back().expression)
                return {};
        }

        // Emit the resolved library operation, not a second arithmetic interpreter.
        // Temporary ports bind each operand to this invocation's expression without
        // modifying the shared graph or leaking its variable names across calls.
        mx::ShaderNode arguments(nullptr, "emission_arguments");
        mx::ShaderNode call(nullptr, "emission_call");
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto* original = node.getInput(i);
            auto* value = arguments.addOutput(original->getName(), original->getType());
            value->setVariable(*inputs[i].expression);
            call.addInput(original->getName(), original->getType())->makeConnection(value);
        }
        const std::string variable = "mx_emission_value_" + std::to_string(m_value_count++);
        call.addOutput(output.getName(), output.getType())->setVariable(variable);
        node.getImplementation().emitFunctionCall(call, m_context, m_calls);
        m_generator.emitFunctionDefinition(node, m_context, m_definitions);
        return expression_value(variable);
    }

    const mx::VariableBlock& m_uniforms;
    const StaticInputQuery& m_static_input_query;
    const GenslangPtShaderGenerator& m_generator;
    mx::GenContext& m_context;
    mx::ShaderStage m_calls;
    mx::ShaderStage m_definitions;
    size_t m_value_count = 0;
};

} // namespace

EmissionQuery build_emission_query(
    const mx::ShaderGraph& graph,
    const mx::VariableBlock& uniforms,
    const StaticInputQuery& static_input_query,
    const GenslangPtShaderGenerator& generator,
    mx::GenContext& context
)
{
    return EmissionQueryBuilder(uniforms, static_input_query, generator, context).build(graph);
}

} // namespace falcor::mtlx::structured
