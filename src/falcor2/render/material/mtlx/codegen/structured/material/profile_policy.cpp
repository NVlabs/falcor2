// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "profile_policy.h"

#include "../graph/static_input_query/static_input_query.h"
#include "../graph/static_input_query/input_static_values.h"
#include "../target/nodes/source_code_node.h"

#include "falcor2/core/error.h"

#include <MaterialXCore/Types.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <functional>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace falcor {
namespace mtlx::structured {

namespace mx = MaterialX;

namespace {

struct ShaderStaticValue {
    std::array<double, 4> values{};
    size_t size = 0;

    bool valid() const { return size > 0; }
};

constexpr double SHADER_STATIC_VALUE_EPSILON = 1.0e-8;

std::vector<double> parse_shader_number_list(std::string value)
{
    std::replace(value.begin(), value.end(), ',', ' ');

    std::stringstream stream(value);
    std::vector<double> result;
    double item = 0.0;
    while (stream >> item)
        result.push_back(item);
    return result;
}

size_t shader_static_component_count(const mx::TypeDesc& type)
{
    const std::string type_name = type.getName();
    if (type_name == "float" || type_name == "integer" || type_name == "boolean")
        return 1;
    if (type_name == "color2" || type_name == "vector2")
        return 2;
    if (type_name == "color3" || type_name == "vector3")
        return 3;
    if (type_name == "color4" || type_name == "vector4")
        return 4;
    return 0;
}

bool supported_static_value_type(const mx::TypeDesc& type)
{
    return shader_static_component_count(type) != 0;
}

std::optional<ShaderStaticValue> parse_shader_static_value(const std::string& value, const mx::TypeDesc& type)
{
    if (value.empty())
        return {};

    if (type == mx::Type::BOOLEAN) {
        std::string lower = value;
        std::transform(
            lower.begin(),
            lower.end(),
            lower.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            }
        );
        if (lower == "true")
            return ShaderStaticValue{{1.0, 0.0, 0.0, 0.0}, 1};
        if (lower == "false")
            return ShaderStaticValue{{0.0, 0.0, 0.0, 0.0}, 1};
    }

    std::vector<double> parsed = parse_shader_number_list(value);
    if (parsed.empty())
        return {};

    size_t count = shader_static_component_count(type);
    if (count == 0)
        count = std::min<size_t>(parsed.size(), 4);

    if (parsed.size() == 1 && count > 1)
        parsed.resize(count, parsed.front());
    if (parsed.size() < count)
        return {};

    ShaderStaticValue result;
    result.size = std::min<size_t>(count, 4);
    for (size_t i = 0; i < result.size; ++i)
        result.values[i] = parsed[i];
    return result;
}

double shader_static_component(const ShaderStaticValue& value, size_t index)
{
    if (value.size == 1)
        return value.values[0];
    if (index < value.size)
        return value.values[index];
    return value.values[value.size - 1];
}

ShaderStaticValue shader_static_scalar_value(double value)
{
    return ShaderStaticValue{{value, 0.0, 0.0, 0.0}, 1};
}

ShaderStaticValue shader_static_binary_value(
    const ShaderStaticValue& lhs,
    const ShaderStaticValue& rhs,
    const std::function<double(double, double)>& operation
)
{
    ShaderStaticValue result;
    result.size = std::max(lhs.size, rhs.size);
    for (size_t i = 0; i < result.size; ++i)
        result.values[i] = operation(shader_static_component(lhs, i), shader_static_component(rhs, i));
    return result;
}

bool shader_static_all_components_equal(const ShaderStaticValue& value, double expected)
{
    if (!value.valid())
        return false;
    for (size_t i = 0; i < value.size; ++i) {
        if (std::abs(value.values[i] - expected) > SHADER_STATIC_VALUE_EPSILON)
            return false;
    }
    return true;
}

bool shader_static_is_zero(const std::optional<ShaderStaticValue>& value)
{
    return value && shader_static_all_components_equal(*value, 0.0);
}

bool shader_static_is_one(const std::optional<ShaderStaticValue>& value)
{
    return value && shader_static_all_components_equal(*value, 1.0);
}

bool shader_input_has_real_connection(const mx::ShaderInput* input)
{
    if (!input || !input->getConnection() || input->isBindInput())
        return false;

    const mx::ShaderNode* source_node = input->getConnection()->getNode();
    return source_node && !source_node->isAGraph();
}

class ShaderStaticEvaluator {
public:
    explicit ShaderStaticEvaluator(const mx::VariableBlock* constants)
        : m_constants(constants)
    {
    }

    std::optional<ShaderStaticValue> input_value(const mx::ShaderNode& node, const std::string& input_name)
    {
        const mx::ShaderInput* input = node.getInput(input_name);
        return input ? port_value(input) : std::optional<ShaderStaticValue>{};
    }

    std::optional<ShaderStaticValue> port_value(const mx::ShaderInput* input)
    {
        if (!input)
            return {};

        const std::string value = input->getValueString();
        const bool has_real_connection = shader_input_has_real_connection(input);
        if (!value.empty() && (input->hasAuthoredValue() || !has_real_connection))
            if (auto parsed = parse_shader_static_value(value, input->getType()))
                return parsed;

        if (const mx::ShaderOutput* connection = has_real_connection ? input->getConnection() : nullptr)
            return node_value(connection->getNode());

        if (auto constant = constant_value(input->getVariable(), input->getType()))
            return constant;
        if (!value.empty())
            return constant_value(value, input->getType());
        return {};
    }

    std::optional<ShaderStaticValue> node_value(const mx::ShaderNode* node)
    {
        if (!node || node->hasClassification(mx::ShaderNode::Classification::CLOSURE))
            return {};

        if (!m_visiting.insert(node).second)
            return {};

        std::optional<ShaderStaticValue> result;
        switch (formal_node_operation(*node)) {
        case FormalNodeOperation::constant:
            if (has_supported_output(*node)) {
                if (has_supported_input(*node, "value"))
                    result = input_value(*node, "value");
                if (!result && has_supported_input(*node, "in"))
                    result = input_value(*node, "in");
            }
            break;
        case FormalNodeOperation::convert:
            if (has_supported_output(*node) && has_supported_input(*node, "in"))
                result = input_value(*node, "in");
            break;
        case FormalNodeOperation::multiply:
            result = multiply_value(*node);
            break;
        case FormalNodeOperation::add:
            result = binary_input_value(*node, "in1", "in2", std::plus<double>{});
            break;
        case FormalNodeOperation::subtract:
            result = binary_input_value(*node, "in1", "in2", std::minus<double>{});
            break;
        case FormalNodeOperation::divide:
            result = divide_value(*node);
            break;
        case FormalNodeOperation::mix:
            result = mix_value(*node);
            break;
        case FormalNodeOperation::clamp:
            result = clamp_value(*node);
            break;
        case FormalNodeOperation::minimum:
            result = binary_input_value(
                *node,
                "in1",
                "in2",
                [](double lhs, double rhs)
                {
                    return std::min(lhs, rhs);
                }
            );
            break;
        case FormalNodeOperation::maximum:
            result = binary_input_value(
                *node,
                "in1",
                "in2",
                [](double lhs, double rhs)
                {
                    return std::max(lhs, rhs);
                }
            );
            break;
        case FormalNodeOperation::absolute_value:
            if (has_supported_output(*node) && has_supported_input(*node, "in")) {
                auto input = input_value(*node, "in");
                if (!input)
                    break;
                ShaderStaticValue value = *input;
                for (size_t i = 0; i < value.size; ++i)
                    value.values[i] = std::abs(value.values[i]);
                result = value;
            }
            break;
        default:
            break;
        }

        m_visiting.erase(node);
        return result;
    }

private:
    std::optional<ShaderStaticValue> binary_input_value(
        const mx::ShaderNode& node,
        const std::string& lhs_name,
        const std::string& rhs_name,
        const std::function<double(double, double)>& operation
    )
    {
        if (!has_supported_output(node) || !has_supported_input(node, lhs_name) || !has_supported_input(node, rhs_name))
            return {};
        auto lhs = input_value(node, lhs_name);
        auto rhs = input_value(node, rhs_name);
        if (!lhs || !rhs)
            return {};
        return shader_static_binary_value(*lhs, *rhs, operation);
    }

    std::optional<ShaderStaticValue> multiply_value(const mx::ShaderNode& node)
    {
        if (!has_supported_output(node) || !has_supported_input(node, "in1") || !has_supported_input(node, "in2"))
            return {};
        auto lhs = input_value(node, "in1");
        auto rhs = input_value(node, "in2");
        if (shader_static_is_zero(lhs) || shader_static_is_zero(rhs))
            return shader_static_scalar_value(0.0);
        if (!lhs || !rhs)
            return {};
        return shader_static_binary_value(
            *lhs,
            *rhs,
            [](double a, double b)
            {
                return a * b;
            }
        );
    }

    std::optional<ShaderStaticValue> divide_value(const mx::ShaderNode& node)
    {
        if (!has_supported_output(node) || !has_supported_input(node, "in1") || !has_supported_input(node, "in2"))
            return {};
        auto lhs = input_value(node, "in1");
        auto rhs = input_value(node, "in2");
        if (!lhs || !rhs)
            return {};
        for (size_t i = 0; i < rhs->size; ++i) {
            if (std::abs(rhs->values[i]) <= SHADER_STATIC_VALUE_EPSILON)
                return {};
        }
        return shader_static_binary_value(
            *lhs,
            *rhs,
            [](double a, double b)
            {
                return a / b;
            }
        );
    }

    std::optional<ShaderStaticValue> mix_value(const mx::ShaderNode& node)
    {
        const bool has_fg_bg = node.getInput("fg") || node.getInput("bg");
        const std::string fg_name = has_fg_bg ? "fg" : "in2";
        const std::string bg_name = has_fg_bg ? "bg" : "in1";

        if (!has_supported_output(node) || !has_supported_input(node, fg_name) || !has_supported_input(node, bg_name)
            || !has_supported_input(node, "mix"))
            return {};

        auto amount = input_value(node, "mix");
        if (shader_static_is_zero(amount))
            return input_value(node, bg_name);
        if (shader_static_is_one(amount))
            return input_value(node, fg_name);

        auto fg = input_value(node, fg_name);
        auto bg = input_value(node, bg_name);
        if (!fg || !bg || !amount)
            return {};

        ShaderStaticValue result;
        result.size = std::max({fg->size, bg->size, amount->size});
        for (size_t i = 0; i < result.size; ++i) {
            const double mix_amount = shader_static_component(*amount, i);
            result.values[i]
                = shader_static_component(*bg, i) * (1.0 - mix_amount) + shader_static_component(*fg, i) * mix_amount;
        }
        return result;
    }

    std::optional<ShaderStaticValue> clamp_value(const mx::ShaderNode& node)
    {
        if (!has_supported_output(node) || !has_supported_input(node, "in") || !has_supported_input(node, "low")
            || !has_supported_input(node, "high"))
            return {};
        auto input = input_value(node, "in");
        auto low = input_value(node, "low");
        auto high = input_value(node, "high");
        if (!input || !low || !high)
            return {};

        ShaderStaticValue result;
        result.size = std::max({input->size, low->size, high->size});
        for (size_t i = 0; i < result.size; ++i) {
            const double lower = shader_static_component(*low, i);
            const double upper = shader_static_component(*high, i);
            result.values[i] = std::min(std::max(shader_static_component(*input, i), lower), upper);
        }
        return result;
    }

    std::unordered_set<const mx::ShaderNode*> m_visiting;
    const mx::VariableBlock* m_constants = nullptr;

    static bool has_supported_output(const mx::ShaderNode& node)
    {
        return node.numOutputs() == 1 && supported_static_value_type(node.getOutput(0)->getType());
    }

    static bool has_supported_input(const mx::ShaderNode& node, const std::string& name)
    {
        const mx::ShaderInput* input = node.getInput(name);
        return input && supported_static_value_type(input->getType());
    }

    std::optional<ShaderStaticValue> constant_value(const std::string& variable, const mx::TypeDesc& type) const
    {
        if (!m_constants || variable.empty())
            return {};
        for (size_t i = 0; i < m_constants->size(); ++i) {
            const mx::ShaderPort* constant = (*m_constants)[i];
            if (!constant || constant->getVariable() != variable)
                continue;
            const std::string value = constant->getValueString();
            return value.empty() ? std::optional<ShaderStaticValue>{} : parse_shader_static_value(value, type);
        }
        return {};
    }
};

std::string_view trim_ascii_space(std::string_view value)
{
    while (!value.empty()
           && (value.front() == ' ' || value.front() == '\t' || value.front() == '\n' || value.front() == '\r'))
        value.remove_prefix(1);
    while (!value.empty()
           && (value.back() == ' ' || value.back() == '\t' || value.back() == '\n' || value.back() == '\r'))
        value.remove_suffix(1);
    return value;
}

MxScatterSelection scatter_selection_from_mode(std::string_view scatter_mode)
{
    scatter_mode = trim_ascii_space(scatter_mode);
    if (scatter_mode == "R" || scatter_mode == "0")
        return MxScatterSelection::static_r;
    if (scatter_mode == "T" || scatter_mode == "1")
        return MxScatterSelection::static_t;
    if (scatter_mode == "RT" || scatter_mode == "2")
        return MxScatterSelection::static_rt;
    return MxScatterSelection::runtime;
}

std::optional<std::string> scatter_mode_from_static_value(const ScalarStaticValue& value)
{
    if (value.kind == ScalarStaticValueKind::KnownZero)
        return "0";
    if (value.kind == ScalarStaticValueKind::KnownOne)
        return "1";
    if (value.kind != ScalarStaticValueKind::KnownConstant || value.value.size != 1)
        return {};

    const float mode = value.value.values[0];
    if (mode == 0.0f || mode == 1.0f || mode == 2.0f)
        return fmt::format("{:.0f}", mode);
    return {};
}

std::string scatter_policy_type(MxScatterSelection selection)
{
    switch (selection) {
    case MxScatterSelection::runtime:
        return "MxScatterRuntimeMode";
    case MxScatterSelection::static_r:
        return "MxScatterStaticRMode";
    case MxScatterSelection::static_t:
        return "MxScatterStaticTMode";
    case MxScatterSelection::static_rt:
        return "MxScatterStaticRTMode";
    }
    FALCOR_UNREACHABLE();
}

std::string refractive_compensation_policy_type(CompensationMode mode)
{
    if (mode == CompensationMode::turquin_analytic)
        return "MxTurquinAnalyticDielectricCompensation";
    if (mode == CompensationMode::no_compensation)
        return "MxNoCompensationDielectricCompensation";
    return "MxTurquinLutDielectricCompensation";
}

std::string microfacet_compensation_policy_type(CompensationMode mode)
{
    if (mode == CompensationMode::turquin_analytic)
        return "MxTurquinAnalyticMicrofacetCompensation";
    if (mode == CompensationMode::no_compensation)
        return "MxNoCompensationMicrofacetCompensation";
    return "MxTurquinLutMicrofacetCompensation";
}

std::optional<std::string>
shader_static_input_value_string_or_default(const mx::ShaderNode& node, const std::string& input_name)
{
    const mx::ShaderInput* input = node.getInput(input_name);
    FALCOR_CHECK(
        input,
        "MTLX node '{}' (impl '{}') is missing nodedef input '{}'.",
        node.getName(),
        node.getImplementation().getName(),
        input_name
    );
    if (shader_input_has_real_connection(input))
        return {};
    return input->getValueString();
}

} // namespace

MxMicrofacetKind microfacet_kind(BsdfKind kind)
{
    if (kind == BsdfKind::dielectric)
        return MxMicrofacetKind::dielectric;
    if (kind == BsdfKind::conductor)
        return MxMicrofacetKind::conductor;
    if (kind == BsdfKind::generalized_schlick)
        return MxMicrofacetKind::generalized_schlick;
    return MxMicrofacetKind::none;
}

std::string scatter_selection_mode_string(MxScatterSelection selection)
{
    switch (selection) {
    case MxScatterSelection::runtime:
        return "RT";
    case MxScatterSelection::static_r:
        return "R";
    case MxScatterSelection::static_t:
        return "T";
    case MxScatterSelection::static_rt:
        return "RT";
    }
    FALCOR_UNREACHABLE();
}

MxMicrofacetSelection select_microfacet(
    const CodeGenDesc& desc,
    BsdfKind kind,
    const mx::ShaderNode* node,
    const mx::VariableBlock* constants,
    const StaticInputQuery* static_input_query
)
{
    MxMicrofacetSelection result;
    result.kind = microfacet_kind(kind);
    result.compensation = desc.compensation_mode;
    if (result.kind == MxMicrofacetKind::none)
        return result;

    result.fresnel = MxFresnelSelection::airy;

    const bool optimize_fresnel = is_set(desc.optimize_graph_flags, OptimizeGraphFlags::fresnel_policy);
    const bool optimize_scatter = is_set(desc.optimize_graph_flags, OptimizeGraphFlags::static_scatter_mode);
    if (!node)
        return result;

    ShaderStaticEvaluator static_eval(constants);
    if (optimize_scatter
        && (result.kind == MxMicrofacetKind::dielectric || result.kind == MxMicrofacetKind::generalized_schlick)) {
        const mx::ShaderInput* scatter_input = node->getInput("scatter_mode");
        FALCOR_CHECK(
            scatter_input,
            "MTLX node '{}' (impl '{}') is missing nodedef input 'scatter_mode'.",
            node->getName(),
            node->getImplementation().getName()
        );

        std::optional<std::string> scatter_value;
        if (scatter_input->getConnection()) {
            if (static_input_query)
                scatter_value = scatter_mode_from_static_value(static_input_query->input_value(*scatter_input));
        } else {
            scatter_value = shader_static_input_value_string_or_default(*node, "scatter_mode");
        }
        const mx::ShaderOutput* scatter_connection = scatter_input->getConnection();
        if (!scatter_value && (!scatter_connection || !has_supported_runtime_value(*scatter_connection))) {
            if (auto static_value = static_eval.input_value(*node, "scatter_mode")) {
                std::ostringstream stream;
                stream << int(std::round(static_value->values[0]));
                scatter_value = stream.str();
            }
        }
        if (scatter_value)
            result.scatter = scatter_selection_from_mode(*scatter_value);
    }

    if (!optimize_fresnel)
        return result;

    const bool thinfilm_is_exact_zero
        = static_input_query && static_input_query->is_exact(*node, "thinfilm_thickness", 0.0f);
    if (!thinfilm_is_exact_zero)
        return result;

    result.fresnel = MxFresnelSelection::standard;
    if (result.kind == MxMicrofacetKind::generalized_schlick) {
        const bool color82_is_exact_one = static_input_query && static_input_query->is_exact(*node, "color82", 1.0f);
        if (!color82_is_exact_one)
            result.fresnel = MxFresnelSelection::color82;
    }
    return result;
}

std::string microfacet_bsdf_type(MxMicrofacetSelection selection)
{
    const std::string scatter_policy = scatter_policy_type(selection.scatter);
    const std::string refractive_compensation = refractive_compensation_policy_type(selection.compensation);

    switch (selection.kind) {
    case MxMicrofacetKind::dielectric: {
        const std::string fresnel_type
            = selection.fresnel == MxFresnelSelection::airy ? "MxDielectricAiryFresnel" : "MxDielectricFresnel";
        return fmt::format(
            "MxCompensatedDielectricBSDF<{}, {}, {}>",
            fresnel_type,
            refractive_compensation,
            scatter_policy
        );
    }
    case MxMicrofacetKind::conductor:
        if (selection.fresnel == MxFresnelSelection::airy)
            return fmt::format(
                "MxCompensatedConductorBSDF<MxConductorAiryFresnel, {}>",
                microfacet_compensation_policy_type(selection.compensation)
            );
        return fmt::format(
            "MxCompensatedConductorBSDF<MxConductorFresnel, {}>",
            microfacet_compensation_policy_type(selection.compensation)
        );
    case MxMicrofacetKind::generalized_schlick: {
        const std::string fresnel_type = selection.fresnel == MxFresnelSelection::airy
            ? "MxSchlickAiryFresnel"
            : (selection.fresnel == MxFresnelSelection::color82 ? "MxSchlickColor82Fresnel" : "MxSchlickFresnel");
        return fmt::format(
            "MxCompensatedGeneralizedSchlickBSDF<{}, {}, {}>",
            fresnel_type,
            refractive_compensation,
            scatter_policy
        );
    }
    case MxMicrofacetKind::none:
        break;
    }
    return "MxNullBSDF";
}

MxMicrofacetSelection common_microfacet_selection(MxMicrofacetSelection a, MxMicrofacetSelection b)
{
    FALCOR_CHECK(a.kind == b.kind, "One MaterialX implementation node resolved to different microfacet families.");
    FALCOR_CHECK(
        a.compensation == b.compensation,
        "One MaterialX implementation node resolved to different global compensation policies."
    );

    if (a.scatter != b.scatter)
        a.scatter = MxScatterSelection::runtime;

    if (a.fresnel != b.fresnel) {
        // Airy is the unspecialized representation. With zero thin-film
        // thickness it reduces to the color82 or standard Fresnel form.
        if (a.fresnel == MxFresnelSelection::airy || b.fresnel == MxFresnelSelection::airy)
            a.fresnel = MxFresnelSelection::airy;
        else
            a.fresnel = MxFresnelSelection::color82;
    }
    return a;
}

MxSheenSelection select_sheen(const CodeGenDesc& desc)
{
    return {desc.zeltner_sheen_coefficients};
}

std::string sheen_bsdf_type(MxSheenSelection selection)
{
    switch (selection.coefficients) {
    case ZeltnerSheenCoefficientMode::analytic_fit:
        return "MxSheenBSDF";
    case ZeltnerSheenCoefficientMode::lut:
        return "MxSheenLutBSDF";
    }
    FALCOR_UNREACHABLE();
}

bool scatter_mode_includes_reflection(std::string_view scatter_mode)
{
    scatter_mode = trim_ascii_space(scatter_mode);
    // MaterialX defines this enum as R,T,RT. Depending on where we read the
    // value in codegen, enum-valued shader inputs may already be normalized to
    // their integer representation, so accept both spellings.
    return scatter_mode.empty() || scatter_mode == "R" || scatter_mode == "RT" || scatter_mode == "0"
        || scatter_mode == "2";
}

bool scatter_mode_includes_transmission(std::string_view scatter_mode)
{
    scatter_mode = trim_ascii_space(scatter_mode);
    return scatter_mode == "T" || scatter_mode == "RT" || scatter_mode == "1" || scatter_mode == "2";
}

bool is_scatter_mode_bsdf(BsdfKind kind)
{
    return kind == BsdfKind::dielectric || kind == BsdfKind::generalized_schlick;
}

} // namespace mtlx::structured
} // namespace falcor
