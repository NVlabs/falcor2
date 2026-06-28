// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "bsdf_mix_layering.h"

#include "falcor2/core/error.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace falcor::materialx::mx139 {
namespace {

class SlangEmitter {
public:
    void line(const std::string& text = {}, bool auto_semicolon = true)
    {
        for (int i = 0; i < m_indent; ++i)
            m_stream << "    ";
        m_stream << text;
        if (auto_semicolon && should_append_semicolon(text))
            m_stream << ";";
        m_stream << "\n";
    }

    void begin(const std::string& text)
    {
        line(text, false);
        line("{");
        ++m_indent;
    }

    void end(const std::string& suffix = {})
    {
        --m_indent;
        line("}" + suffix);
    }

    std::string str() const { return m_stream.str(); }

private:
    static bool starts_with(const std::string& text, const std::string& prefix) { return text.rfind(prefix, 0) == 0; }

    static bool should_append_semicolon(const std::string& text)
    {
        if (text.empty())
            return false;
        if (text.find("= {}") != std::string::npos)
            return true;
        const char last = text.back();
        if (last == ';' || last == '{' || last == '}' || last == ':')
            return false;
        if (starts_with(text, "//") || starts_with(text, "#"))
            return false;
        if (starts_with(text, "switch "))
            return false;
        if (starts_with(text, "public property ") || starts_with(text, "get "))
            return false;
        if (starts_with(text, "public __init") && text.find('{') != std::string::npos)
            return false;
        return true;
    }

    std::ostringstream m_stream;
    int m_indent = 0;
};

bool is_bsdf_ref(ClosureRef ref)
{
    FALCOR_CHECK(!ref.is_none(), "Mx139 bsdf_mix layering cannot encode a null closure reference.");
    return ref.is_bsdf();
}

int ref_index(ClosureRef ref)
{
    if (ref.is_bsdf())
        return static_cast<int>(ref.bsdf_index());
    FALCOR_CHECK(ref.is_combiner(), "Mx139 bsdf_mix layering cannot encode a null closure reference.");
    return static_cast<int>(ref.combiner_index());
}

std::string bsdf_field(int index)
{
    return "bsdf" + std::to_string(index);
}

std::string bsdf_param(int index)
{
    return bsdf_field(index) + "_";
}

std::string bsdf_weight_field(int index)
{
    return "bsdf_weights[" + std::to_string(index) + "]";
}

std::string bsdf_frame_field(int index)
{
    return "bsdf_frames[" + std::to_string(index) + "]";
}

std::string combiner_weight_local(int index)
{
    return "combiner_weight[" + std::to_string(index) + "]";
}

std::string combiner_mix_local(int index)
{
    return "mix[" + std::to_string(index) + "]";
}

std::string combiner_inverse_mix_local(int index)
{
    return "(float3(1.0) - " + combiner_mix_local(index) + ")";
}

std::string combiner_active_local(int index)
{
    return "combiner_active[" + std::to_string(index) + "]";
}

std::string combiner_branch0_sample_albedo_local(int index)
{
    return "branch0_sample_albedo[" + std::to_string(index) + "]";
}

std::string combiner_branch1_sample_albedo_local(int index)
{
    return "branch1_sample_albedo[" + std::to_string(index) + "]";
}

std::string combiner_branch0_eval_scale_local(int index)
{
    return "branch0_eval_scale[" + std::to_string(index) + "]";
}

std::string combiner_branch1_eval_scale_local(int index)
{
    return "branch1_eval_scale[" + std::to_string(index) + "]";
}

std::string combiner_branch0_probability_local(int index)
{
    return "combiner_branch0_probability[" + std::to_string(index) + "]";
}

std::string combiner_mode_name(MxFlatCombinerMode mode)
{
    switch (mode) {
    case MxFlatCombinerMode::layer:
        return "layer";
    case MxFlatCombinerMode::mix:
        return "mix";
    case MxFlatCombinerMode::add:
        return "add";
    }
    FALCOR_UNREACHABLE();
}

std::string combiner_sample_probability_branch0_local(const MxFlatCombinerDesc& combiner)
{
    return "combiner_" + std::to_string(combiner.index) + "_" + combiner_mode_name(combiner.mode)
        + "_branch0_probability";
}

std::string combiner_sample_probability_branch1_local(const MxFlatCombinerDesc& combiner)
{
    return "combiner_" + std::to_string(combiner.index) + "_" + combiner_mode_name(combiner.mode)
        + "_branch1_probability";
}

std::string combiner_summary_local(int index, const std::string& kind)
{
    return "combiner_" + kind + "[" + std::to_string(index) + "]";
}

std::string ref_value(ClosureRef ref, const std::string& kind)
{
    const int index = ref_index(ref);
    if (is_bsdf_ref(ref))
        return "bsdf_summaries[" + std::to_string(index) + "]." + kind;
    return combiner_summary_local(index, kind);
}

std::string ref_albedo(ClosureRef ref)
{
    return ref_value(ref, "albedo");
}

std::string ref_opacity(ClosureRef ref)
{
    return ref_value(ref, "opacity");
}

std::string ref_reflection(ClosureRef ref)
{
    return ref_value(ref, "reflection");
}

std::string ref_transmission(ClosureRef ref)
{
    return ref_value(ref, "transmission");
}

std::string ref_roughness(ClosureRef ref)
{
    return ref_value(ref, "roughness");
}

std::string ref_ior(ClosureRef ref)
{
    return ref_value(ref, "ior");
}

std::string layer_pass_through_albedo_mode_expr(MxFlatLayerPassThroughAlbedoMode mode)
{
    switch (mode) {
    case MxFlatLayerPassThroughAlbedoMode::scattering:
        return "flat::LayerPassThroughAlbedoMode.scattering";
    case MxFlatLayerPassThroughAlbedoMode::reflection:
        return "flat::LayerPassThroughAlbedoMode.reflection";
    case MxFlatLayerPassThroughAlbedoMode::transmissive_unity:
        return "flat::LayerPassThroughAlbedoMode.transmissive_unity";
    }
    FALCOR_UNREACHABLE();
}

std::string layer_pass_through_transmission_mode_expr(MxFlatLayerPassThroughTransmissionMode mode)
{
    switch (mode) {
    case MxFlatLayerPassThroughTransmissionMode::physical:
        return "flat::LayerPassThroughTransmissionMode.physical";
    case MxFlatLayerPassThroughTransmissionMode::zero:
        return "flat::LayerPassThroughTransmissionMode.zero";
    case MxFlatLayerPassThroughTransmissionMode::one:
        return "flat::LayerPassThroughTransmissionMode.one";
    case MxFlatLayerPassThroughTransmissionMode::absorption:
        return "flat::LayerPassThroughTransmissionMode.absorption";
    case MxFlatLayerPassThroughTransmissionMode::absorption_unless_transmissive:
        return "flat::LayerPassThroughTransmissionMode.absorption_unless_transmissive";
    case MxFlatLayerPassThroughTransmissionMode::physical_unless_transmissive:
        return "flat::LayerPassThroughTransmissionMode.physical_unless_transmissive";
    }
    FALCOR_UNREACHABLE();
}

std::string bsdf_layer_pass_through_transmissive_expr(const MxFlatBsdfDesc& bsdf)
{
    if (!bsdf.uses_transmission_tint_predicate)
        return "false";
    return "(mx_hmax(" + bsdf.field_name + ".transmission_tint) > 0.0)";
}

std::string roughness_lerp(const std::string& a, const std::string& b, const std::string& t)
{
    return "RoughnessInformation(lerp(" + a + ".roughness, " + b + ".roughness, " + t + "), lerp(" + a + ".scratch, "
        + b + ".scratch, " + t + "))";
}

std::string multiply_probability_term(const std::string& probability, const std::string& factor)
{
    if (probability == "1.0")
        return factor;
    if (probability.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
        == std::string::npos)
        return probability + " * " + factor;
    return "(" + probability + ") * " + factor;
}

std::string multiply_eval_scale_term(const std::string& scale, const std::string& factor)
{
    if (scale.empty() || scale == "float3(1.0)")
        return factor;
    if (scale.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == std::string::npos)
        return scale + " * " + factor;
    return "(" + scale + ") * " + factor;
}

std::string direct_spine_local(
    const MxFlatCombinerDesc& combiner,
    bool branch0_child,
    bool branch1_child,
    bool branch0_branch,
    int use_index
)
{
    const bool only_one_child_continues = branch0_child != branch1_child;
    std::string result = "combiner_" + std::to_string(combiner.index) + "_" + combiner_mode_name(combiner.mode) + "_";
    if (!only_one_child_continues)
        result += branch0_branch ? "branch0_" : "branch1_";
    result += "path_probability";
    if (use_index != 0)
        result += "_" + std::to_string(use_index);
    return result;
}

std::string
eval_scale_spine_local(int combiner_index, bool branch0_child, bool branch1_child, bool branch0_branch, int use_index)
{
    const bool only_one_child_continues = branch0_child != branch1_child;
    std::string result = "eval_spine_";
    if (!only_one_child_continues)
        result += branch0_branch ? "branch0_" : "branch1_";
    result += std::to_string(combiner_index);
    if (use_index != 0)
        result += "_" + std::to_string(use_index);
    return result;
}

void collect_bsdf_sample_probability_terms(
    SlangEmitter& e,
    const MxFlatRootDesc& desc,
    ClosureRef ref,
    const std::string& probability,
    std::vector<std::vector<std::string>>& bsdf_terms,
    std::vector<bool>& visiting_combiners,
    std::vector<int>& spine_uses
)
{
    if (is_bsdf_ref(ref)) {
        const int bsdf_index = ref_index(ref);
        FALCOR_CHECK(
            bsdf_index >= 0 && size_t(bsdf_index) < bsdf_terms.size(),
            "Mx139 bsdf_mix sample probability references invalid BSDF {}.",
            bsdf_index
        );
        bsdf_terms[size_t(bsdf_index)].push_back(probability);
        return;
    }

    const int combiner_index = ref_index(ref);
    FALCOR_CHECK(
        combiner_index >= 0 && size_t(combiner_index) < desc.combiners.size(),
        "Mx139 bsdf_mix sample probability references invalid combiner {}.",
        combiner_index
    );
    FALCOR_CHECK(
        !visiting_combiners[size_t(combiner_index)],
        "Mx139 bsdf_mix sample probability graph contains a cycle."
    );
    visiting_combiners[size_t(combiner_index)] = true;

    const MxFlatCombinerDesc& combiner = desc.combiners[size_t(combiner_index)];
    FALCOR_CHECK(combiner.index == combiner_index, "Mx139 bsdf_mix combiner index mismatch.");
    const bool branch0_child_continues = !is_bsdf_ref(combiner.branch0_ref);
    const bool branch1_child_continues = !is_bsdf_ref(combiner.branch1_ref);

    const auto descend = [&](ClosureRef child_ref, const std::string& factor, bool branch0_branch)
    {
        std::string child_probability = multiply_probability_term(probability, factor);
        if (!is_bsdf_ref(child_ref)) {
            const std::string spine = direct_spine_local(
                combiner,
                branch0_child_continues,
                branch1_child_continues,
                branch0_branch,
                spine_uses[size_t(combiner_index)]++
            );
            e.line("float " + spine + " = " + child_probability);
            child_probability = spine;
        }
        collect_bsdf_sample_probability_terms(
            e,
            desc,
            child_ref,
            child_probability,
            bsdf_terms,
            visiting_combiners,
            spine_uses
        );
    };

    descend(combiner.branch0_ref, combiner_sample_probability_branch0_local(combiner), true);
    descend(combiner.branch1_ref, combiner_sample_probability_branch1_local(combiner), false);

    visiting_combiners[size_t(combiner_index)] = false;
}

void collect_bsdf_eval_scale_terms(
    SlangEmitter& e,
    const MxFlatRootDesc& desc,
    ClosureRef ref,
    const std::string& scale,
    std::vector<std::vector<std::string>>& bsdf_terms,
    std::vector<bool>& visiting_combiners,
    std::vector<int>& spine_uses
)
{
    if (is_bsdf_ref(ref)) {
        const int bsdf_index = ref_index(ref);
        FALCOR_CHECK(
            bsdf_index >= 0 && size_t(bsdf_index) < bsdf_terms.size(),
            "Mx139 bsdf_mix eval scale references invalid BSDF {}.",
            bsdf_index
        );
        bsdf_terms[size_t(bsdf_index)].push_back(scale.empty() ? "float3(1.0)" : scale);
        return;
    }

    const int combiner_index = ref_index(ref);
    FALCOR_CHECK(
        combiner_index >= 0 && size_t(combiner_index) < desc.combiners.size(),
        "Mx139 bsdf_mix eval scale references invalid combiner {}.",
        combiner_index
    );
    FALCOR_CHECK(!visiting_combiners[size_t(combiner_index)], "Mx139 bsdf_mix eval scale graph contains a cycle.");
    visiting_combiners[size_t(combiner_index)] = true;

    const MxFlatCombinerDesc& combiner = desc.combiners[size_t(combiner_index)];
    FALCOR_CHECK(combiner.index == combiner_index, "Mx139 bsdf_mix combiner index mismatch.");
    const bool branch0_child_continues = !is_bsdf_ref(combiner.branch0_ref);
    const bool branch1_child_continues = !is_bsdf_ref(combiner.branch1_ref);

    const auto descend = [&](ClosureRef child_ref, const std::string& factor, bool branch0_branch)
    {
        std::string child_scale = multiply_eval_scale_term(scale, factor);
        if (!is_bsdf_ref(child_ref)) {
            const std::string spine = eval_scale_spine_local(
                combiner_index,
                branch0_child_continues,
                branch1_child_continues,
                branch0_branch,
                spine_uses[size_t(combiner_index)]++
            );
            e.line("float3 " + spine + " = " + child_scale);
            child_scale = spine;
        }
        collect_bsdf_eval_scale_terms(e, desc, child_ref, child_scale, bsdf_terms, visiting_combiners, spine_uses);
    };

    descend(combiner.branch0_ref, combiner_branch0_eval_scale_local(combiner_index), true);
    descend(combiner.branch1_ref, combiner_branch1_eval_scale_local(combiner_index), false);

    visiting_combiners[size_t(combiner_index)] = false;
}

std::string sum_terms(const std::vector<std::string>& terms)
{
    FALCOR_CHECK(!terms.empty(), "Cannot sum empty Mx139 bsdf_mix term list.");
    std::string result;
    for (size_t i = 0; i < terms.size(); ++i) {
        if (i != 0)
            result += " + ";
        result += terms[i];
    }
    return result;
}

void emit_direct_sample_probabilities(SlangEmitter& e, const MxFlatRootDesc& desc)
{
    for (const MxFlatCombinerDesc& combiner : desc.combiners) {
        e.line(
            "float " + combiner_sample_probability_branch0_local(combiner) + " = "
            + combiner_active_local(combiner.index) + " * " + combiner_branch0_probability_local(combiner.index)
        );
        e.line(
            "float " + combiner_sample_probability_branch1_local(combiner) + " = "
            + combiner_active_local(combiner.index) + " - " + combiner_sample_probability_branch0_local(combiner)
        );
    }

    std::vector<std::vector<std::string>> bsdf_terms(desc.bsdfs.size());
    std::vector<bool> visiting_combiners(desc.combiners.size(), false);
    std::vector<int> spine_uses(desc.combiners.size(), 0);
    collect_bsdf_sample_probability_terms(e, desc, desc.root_ref, "1.0", bsdf_terms, visiting_combiners, spine_uses);

    for (size_t i = 0; i < bsdf_terms.size(); ++i) {
        if (!bsdf_terms[i].empty()) {
            const std::string bsdf_probability = "bsdf_" + std::to_string(i) + "_sample_probability";
            e.line("float " + bsdf_probability + " = max(" + sum_terms(bsdf_terms[i]) + ", 0.0)");
            e.line("bsdf_sample_probabilities[" + std::to_string(i) + "] = " + bsdf_probability);
        }
    }
}

void emit_final_bsdf_weights(SlangEmitter& e, const MxFlatRootDesc& desc)
{
    std::vector<std::vector<std::string>> bsdf_terms(desc.bsdfs.size());
    std::vector<bool> visiting_combiners(desc.combiners.size(), false);
    std::vector<int> spine_uses(desc.combiners.size(), 0);
    collect_bsdf_eval_scale_terms(e, desc, desc.root_ref, "", bsdf_terms, visiting_combiners, spine_uses);

    for (size_t i = 0; i < bsdf_terms.size(); ++i) {
        if (!bsdf_terms[i].empty())
            e.line("bsdf_weights[" + std::to_string(i) + "] *= " + sum_terms(bsdf_terms[i]));
        else
            e.line("bsdf_weights[" + std::to_string(i) + "] = float3(0.0)");
    }
}

void emit_sample_probabilities(SlangEmitter& e, const MxFlatRootDesc& desc)
{
    emit_direct_sample_probabilities(e, desc);
}

void emit_bsdf_dispatch(SlangEmitter& e, const MxFlatRootDesc& desc)
{
    e.begin(
        "bool sample_bsdf<S : ISampleGenerator>(int bsdf_index, const float3 wi, out float3 "
        "wo, out float pdf, out float3 weight, out LobeTypes lobe_types, inout S sg, const BSDFContext bc)"
    );
    e.line("switch (bsdf_index)");
    e.line("{");
    for (size_t i = 0; i < desc.bsdfs.size(); ++i) {
        const int bsdf_index = int(i);
        e.line(
            "case " + std::to_string(i) + ": return flat::sample_bsdf(" + desc.bsdfs[i].field_name + ", "
            + bsdf_weight_field(bsdf_index) + ", " + bsdf_frame_field(bsdf_index)
            + ", wi, wo, pdf, weight, lobe_types, sg, bc);"
        );
    }
    e.line("default: break;");
    e.line("}");
    e.line("wo = float3(0.0)");
    e.line("pdf = 0.0");
    e.line("weight = float3(0.0)");
    e.line("lobe_types = LobeTypes::none");
    e.line("return false");
    e.end();
}

void emit_eval_pdf(SlangEmitter& e, const MxFlatRootDesc& desc)
{
    e.begin("public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)");
    e.line("float result = 0.0");
    for (size_t i = 0; i < desc.bsdfs.size(); ++i) {
        const int bsdf_index = int(i);
        e.line(
            "result += bsdf_sample_probabilities[" + std::to_string(i) + "] * flat::eval_bsdf_pdf("
            + desc.bsdfs[i].field_name + ", " + bsdf_weight_field(bsdf_index) + ", " + bsdf_frame_field(bsdf_index)
            + ", wi, wo, bc)"
        );
    }
    e.line("return result");
    e.end();
}

void emit_cdf_sample_bsdf_selector(SlangEmitter& e, const MxFlatRootDesc& desc)
{
    e.begin(
        "bool select_cdf_sample_bsdf<S : ISampleGenerator>(inout S sg, out int bsdf_index, out float path_probability)"
    );
    e.line("bsdf_index = 0");
    e.line("path_probability = 0.0");
    if (desc.bsdfs.size() == 1) {
        e.line("path_probability = bsdf_sample_probabilities[0]");
        e.line("return path_probability > 0.0");
        e.end();
        return;
    }
    e.line("float selector = next_float(sg)");
    e.line("float cdf = 0.0");
    e.line("[ForceUnroll]", false);
    e.begin("for (int i = 0; i < " + std::to_string(desc.bsdfs.size()) + "; ++i)");
    e.line("float p = bsdf_sample_probabilities[i]");
    e.line("if (p <= 0.0)", false);
    e.line("    continue");
    e.line("cdf += p");
    e.begin("if (selector < cdf)");
    e.line("bsdf_index = i");
    e.line("path_probability = p");
    e.line("return true");
    e.end();
    e.end();
    e.line("return false");
    e.end();
}

void emit_sample(SlangEmitter& e)
{
    e.begin(
        "public bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float "
        "pdf, out float3 weight, out LobeTypes lobe_types, inout S sg, const BSDFContext bc)"
    );
    e.line("// MX139 bsdf_mix sample selector: direct CDF over final BSDF probabilities.", false);
    e.line("wo = float3(0.0)");
    e.line("pdf = 0.0");
    e.line("weight = float3(0.0)");
    e.line("lobe_types = LobeTypes::none");
    e.line();
    e.line("int bsdf_index = 0");
    e.line("float path_probability = 0.0");
    e.line("if (!select_cdf_sample_bsdf(sg, bsdf_index, path_probability)) return false");
    e.line("float branch_pdf = 0.0");
    e.line("float3 branch_weight = float3(0.0)");
    e.line(
        "bool valid = sample_bsdf(bsdf_index, wi, wo, branch_pdf, branch_weight, lobe_types, sg, "
        "bc)"
    );
    e.begin("if (!valid)");
    e.line("wo = float3(0.0)");
    e.line("pdf = 0.0");
    e.line("weight = float3(0.0)");
    e.line("lobe_types = LobeTypes::none");
    e.line("return false");
    e.end();
    e.begin("if ((uint(lobe_types) & uint(LobeTypes::delta)) != 0)");
    e.line("pdf = path_probability * branch_pdf");
    e.line("weight = branch_weight / path_probability");
    e.line("return true");
    e.end();
    e.line("pdf = eval_pdf(wi, wo, bc)");
    e.begin("if (pdf <= 0.0)");
    e.line("wo = float3(0.0)");
    e.line("pdf = 0.0");
    e.line("weight = float3(0.0)");
    e.line("lobe_types = LobeTypes::none");
    e.line("return false");
    e.end();
    e.line("weight = eval(wi, wo, sg, bc) / pdf");
    e.line("return true");
    e.end();
}

} // namespace

std::string emit_bsdf_mix_root_bsdf(const MxFlatRootDesc& desc)
{
    FALCOR_CHECK(!desc.type_name.empty(), "Mx139 bsdf_mix root type name cannot be empty.");
    FALCOR_CHECK(!desc.bsdfs.empty(), "Mx139 bsdf_mix root requires at least one BSDF.");

    SlangEmitter e;
    const size_t bsdf_count = std::max<size_t>(desc.bsdfs.size(), 1);
    const size_t combiner_count = std::max<size_t>(desc.combiners.size(), 1);

    e.line("// Mx139 bsdf_mix layering root generated from the MaterialX closure tree.");
    e.begin("public struct " + desc.type_name + " : IBSDF");
    for (const MxFlatBsdfDesc& bsdf : desc.bsdfs)
        e.line("public " + bsdf.type_name + " " + bsdf.field_name + " = {}");
    e.line("public float3 bsdf_weights[" + std::to_string(bsdf_count) + "] = {}");
    e.line("public float3 bsdf_transmission_scales[" + std::to_string(bsdf_count) + "] = {}");
    e.line("public Frame bsdf_frames[" + std::to_string(bsdf_count) + "] = {}");
    e.line("public float bsdf_sample_probabilities[" + std::to_string(bsdf_count) + "] = {}");
    e.line("public float3 root_albedo = float3(0.0)");
    e.line("public float3 root_reflection = float3(0.0)");
    e.line("public float3 root_transmission = float3(0.0)");
    e.line("public RoughnessInformation root_roughness = RoughnessInformation()");
    e.line("public float3 root_ior_as_reflectance = float3(0.0)");
    e.line();
    e.line("public __init() { }");

    std::string constructor = "public __init(";
    bool first = true;
    auto append_param = [&](const std::string& param)
    {
        if (!first)
            constructor += ", ";
        first = false;
        constructor += param;
    };
    for (size_t i = 0; i < desc.bsdfs.size(); ++i)
        append_param(desc.bsdfs[i].type_name + " " + bsdf_param(int(i)));
    append_param("const float3 bsdf_weights_[" + std::to_string(bsdf_count) + "]");
    append_param("const float3 bsdf_transmission_scales_[" + std::to_string(bsdf_count) + "]");
    append_param("const Frame bsdf_frames_[" + std::to_string(bsdf_count) + "]");
    if (!desc.combiners.empty()) {
        append_param("const float3 layer_weights_[" + std::to_string(combiner_count) + "]");
        append_param("const float3 mix_values_[" + std::to_string(combiner_count) + "]");
    }
    append_param("float3 wi");
    constructor += ")";
    e.begin(constructor);

    e.begin("// Store BSDF closures and constructor inputs.");
    for (size_t i = 0; i < desc.bsdfs.size(); ++i)
        e.line(desc.bsdfs[i].field_name + " = " + bsdf_param(int(i)));
    e.line("[ForceUnroll]", false);
    e.begin("for (int i = 0; i < " + std::to_string(bsdf_count) + "; ++i)");
    e.line("bsdf_weights[i] = max(bsdf_weights_[i], float3(0.0))");
    e.line("bsdf_transmission_scales[i] = max(bsdf_transmission_scales_[i], float3(0.0))");
    e.line("bsdf_frames[i] = bsdf_frames_[i]");
    e.end();
    e.end();

    e.line();
    e.line("flat::BsdfSummary bsdf_summaries[" + std::to_string(bsdf_count) + "] = {}");
    e.begin("// Read all BSDF closure values before flattening the closure tree.");
    for (size_t i = 0; i < desc.bsdfs.size(); ++i) {
        const int bsdf_index = int(i);
        const MxFlatBsdfDesc& bsdf = desc.bsdfs[i];
        const std::string local_wi = "bsdf_local_wi_" + std::to_string(i);
        const std::string eval_albedo = "bsdf_eval_albedo_" + std::to_string(i);
        const std::string layer_pass_through_transmissive = "layer_pass_through_transmissive_" + std::to_string(i);
        const std::string compatibility_albedo = "compatibility_albedo_" + std::to_string(i);
        const std::string albedo_mode = layer_pass_through_albedo_mode_expr(bsdf.layer_pass_through_albedo_mode);
        const std::string transmission_mode
            = layer_pass_through_transmission_mode_expr(bsdf.layer_pass_through_transmission_mode);
        e.line("float3 " + local_wi + " = " + bsdf_frame_field(bsdf_index) + ".to_local(wi)");
        e.line("AlbedoContributions " + eval_albedo + " = " + bsdf.field_name + ".eval_albedo(" + local_wi + ")");
        e.line("bool " + layer_pass_through_transmissive + " = " + bsdf_layer_pass_through_transmissive_expr(bsdf));
        e.line(
            "flat::BsdfCompatibilityAlbedo " + compatibility_albedo + " = flat::bsdf_compatibility_albedo("
            + eval_albedo + ", " + albedo_mode + ", " + transmission_mode + ", " + layer_pass_through_transmissive + ")"
        );
        e.line(
            "bsdf_summaries[" + std::to_string(i) + "] = flat::summarize_bsdf(" + bsdf.field_name + ", "
            + bsdf_weight_field(bsdf_index) + ", " + local_wi + ", " + compatibility_albedo + ")"
        );
    }
    e.end();

    e.line();
    if (!desc.combiners.empty()) {
        e.line("// Allocate bsdf_mix combiner state.");
        e.line("float3 combiner_weight[" + std::to_string(combiner_count) + "] = {}");
        e.line("float3 mix[" + std::to_string(combiner_count) + "] = {}");
        e.line("float3 branch0_sample_albedo[" + std::to_string(combiner_count) + "] = {}");
        e.line("float3 branch1_sample_albedo[" + std::to_string(combiner_count) + "] = {}");
        e.line("float3 branch0_eval_scale[" + std::to_string(combiner_count) + "] = {}");
        e.line("float3 branch1_eval_scale[" + std::to_string(combiner_count) + "] = {}");
        e.line("float3 combiner_albedo[" + std::to_string(combiner_count) + "] = {}");
        e.line("float3 combiner_opacity[" + std::to_string(combiner_count) + "] = {}");
        e.line("float3 combiner_reflection[" + std::to_string(combiner_count) + "] = {}");
        e.line("float3 combiner_transmission[" + std::to_string(combiner_count) + "] = {}");
        e.line("RoughnessInformation combiner_roughness[" + std::to_string(combiner_count) + "] = {}");
        e.line("float3 combiner_ior[" + std::to_string(combiner_count) + "] = {}");
        e.line("float combiner_active[" + std::to_string(combiner_count) + "] = {}");
        e.line("float combiner_branch0_probability[" + std::to_string(combiner_count) + "] = {}");

        e.begin("// Initialize combiner controls.");
        e.line("[ForceUnroll]", false);
        e.begin("for (int i = 0; i < " + std::to_string(combiner_count) + "; ++i)");
        e.line("combiner_weight[i] = max(layer_weights_[i], float3(0.0))");
        e.line("combiner_active[i] = mx_hmax(combiner_weight[i]) > 0.0 ? 1.0 : 0.0");
        e.line("mix[i] = saturate(mix_values_[i])");
        e.end();
        e.end();

        e.begin("// Reduce closure tree into combiner summaries and branch scales.");
    }
    for (const MxFlatCombinerDesc& combiner : desc.combiners) {
        const int i = combiner.index;

        if (combiner.mode == MxFlatCombinerMode::layer) {
            e.line(
                "float3 base_scale_" + std::to_string(i) + " = saturate(float3(1.0) - "
                + ref_opacity(combiner.branch0_ref) + ")"
            );
            e.line(combiner_branch0_sample_albedo_local(i) + " = " + ref_albedo(combiner.branch0_ref));
            e.line(
                combiner_branch1_sample_albedo_local(i) + " = base_scale_" + std::to_string(i) + " * "
                + ref_albedo(combiner.branch1_ref)
            );
            e.line(combiner_branch0_eval_scale_local(i) + " = " + combiner_weight_local(i));
            e.line(
                combiner_branch1_eval_scale_local(i) + " = " + combiner_weight_local(i) + " * base_scale_"
                + std::to_string(i)
            );
            e.line(
                combiner_summary_local(i, "albedo") + " = " + combiner_weight_local(i) + " * ("
                + ref_albedo(combiner.branch0_ref) + " + base_scale_" + std::to_string(i) + " * "
                + ref_albedo(combiner.branch1_ref) + ")"
            );
            e.line(
                combiner_summary_local(i, "opacity") + " = saturate(" + combiner_weight_local(i) + " * "
                + ref_opacity(combiner.branch1_ref) + ")"
            );
            e.line(
                combiner_summary_local(i, "reflection") + " = " + combiner_weight_local(i) + " * ("
                + ref_reflection(combiner.branch0_ref) + " + base_scale_" + std::to_string(i) + " * "
                + ref_reflection(combiner.branch1_ref) + ")"
            );
            e.line(
                combiner_summary_local(i, "transmission") + " = " + combiner_weight_local(i) + " * base_scale_"
                + std::to_string(i) + " * " + ref_transmission(combiner.branch1_ref)
            );
            e.line(
                combiner_summary_local(i, "reflection") + " = max(" + combiner_summary_local(i, "albedo") + " - "
                + combiner_summary_local(i, "transmission") + ", float3(0.0))"
            );
            e.line(combiner_summary_local(i, "roughness") + " = " + ref_roughness(combiner.branch0_ref));
            e.line(
                combiner_summary_local(i, "ior") + " = " + combiner_weight_local(i) + " * "
                + ref_ior(combiner.branch0_ref)
            );
        } else if (combiner.mode == MxFlatCombinerMode::mix) {
            e.line(
                combiner_branch0_sample_albedo_local(i) + " = " + combiner_inverse_mix_local(i) + " * "
                + ref_albedo(combiner.branch0_ref)
            );
            e.line(
                combiner_branch1_sample_albedo_local(i) + " = " + combiner_mix_local(i) + " * "
                + ref_albedo(combiner.branch1_ref)
            );
            e.line(
                combiner_branch0_eval_scale_local(i) + " = " + combiner_weight_local(i) + " * "
                + combiner_inverse_mix_local(i)
            );
            e.line(
                combiner_branch1_eval_scale_local(i) + " = " + combiner_weight_local(i) + " * " + combiner_mix_local(i)
            );
            e.line(
                combiner_summary_local(i, "albedo") + " = " + combiner_weight_local(i) + " * ("
                + combiner_inverse_mix_local(i) + " * " + ref_albedo(combiner.branch0_ref) + " + "
                + combiner_mix_local(i) + " * " + ref_albedo(combiner.branch1_ref) + ")"
            );
            e.line(
                combiner_summary_local(i, "opacity") + " = saturate(" + combiner_weight_local(i) + " * ("
                + combiner_inverse_mix_local(i) + " * " + ref_opacity(combiner.branch0_ref) + " + "
                + combiner_mix_local(i) + " * " + ref_opacity(combiner.branch1_ref) + "))"
            );
            e.line(
                combiner_summary_local(i, "reflection") + " = " + combiner_weight_local(i) + " * ("
                + combiner_inverse_mix_local(i) + " * " + ref_reflection(combiner.branch0_ref) + " + "
                + combiner_mix_local(i) + " * " + ref_reflection(combiner.branch1_ref) + ")"
            );
            e.line(
                combiner_summary_local(i, "transmission") + " = " + combiner_weight_local(i) + " * ("
                + combiner_inverse_mix_local(i) + " * " + ref_transmission(combiner.branch0_ref) + " + "
                + combiner_mix_local(i) + " * " + ref_transmission(combiner.branch1_ref) + ")"
            );
            e.line(
                combiner_summary_local(i, "reflection") + " = max(" + combiner_summary_local(i, "albedo") + " - "
                + combiner_summary_local(i, "transmission") + ", float3(0.0))"
            );
            e.line(
                "float mix_branch1_p_" + std::to_string(i) + " = flat::balance_probability("
                + combiner_branch1_sample_albedo_local(i) + ", " + combiner_branch0_sample_albedo_local(i) + ")"
            );
            e.line(
                combiner_summary_local(i, "roughness") + " = "
                + roughness_lerp(
                    ref_roughness(combiner.branch0_ref),
                    ref_roughness(combiner.branch1_ref),
                    "mix_branch1_p_" + std::to_string(i)
                )
            );
            e.line(
                combiner_summary_local(i, "ior") + " = " + combiner_weight_local(i) + " * ("
                + combiner_inverse_mix_local(i) + " * " + ref_ior(combiner.branch0_ref) + " + " + combiner_mix_local(i)
                + " * " + ref_ior(combiner.branch1_ref) + ")"
            );
        } else {
            e.line(combiner_branch0_sample_albedo_local(i) + " = " + ref_albedo(combiner.branch0_ref));
            e.line(combiner_branch1_sample_albedo_local(i) + " = " + ref_albedo(combiner.branch1_ref));
            e.line(combiner_branch0_eval_scale_local(i) + " = " + combiner_weight_local(i));
            e.line(combiner_branch1_eval_scale_local(i) + " = " + combiner_weight_local(i));
            e.line(
                combiner_summary_local(i, "albedo") + " = " + combiner_weight_local(i) + " * ("
                + ref_albedo(combiner.branch0_ref) + " + " + ref_albedo(combiner.branch1_ref) + ")"
            );
            e.line(
                combiner_summary_local(i, "opacity") + " = saturate(" + combiner_weight_local(i) + " * saturate("
                + ref_opacity(combiner.branch0_ref) + " + " + ref_opacity(combiner.branch1_ref) + "))"
            );
            e.line(
                combiner_summary_local(i, "reflection") + " = " + combiner_weight_local(i) + " * saturate("
                + ref_reflection(combiner.branch0_ref) + " + " + ref_reflection(combiner.branch1_ref) + ")"
            );
            e.line(
                combiner_summary_local(i, "transmission") + " = " + combiner_weight_local(i) + " * saturate("
                + ref_transmission(combiner.branch0_ref) + " + " + ref_transmission(combiner.branch1_ref) + ")"
            );
            e.line(
                combiner_summary_local(i, "reflection") + " = max(" + combiner_summary_local(i, "albedo") + " - "
                + combiner_summary_local(i, "transmission") + ", float3(0.0))"
            );
            e.line(
                "float add_p_" + std::to_string(i) + " = flat::balance_probability(" + ref_albedo(combiner.branch0_ref)
                + ", " + ref_albedo(combiner.branch1_ref) + ")"
            );
            e.line(
                combiner_summary_local(i, "roughness") + " = "
                + roughness_lerp(
                    ref_roughness(combiner.branch1_ref),
                    ref_roughness(combiner.branch0_ref),
                    "add_p_" + std::to_string(i)
                )
            );
            e.line(
                combiner_summary_local(i, "ior") + " = " + combiner_weight_local(i) + " * saturate("
                + ref_ior(combiner.branch0_ref) + " + " + ref_ior(combiner.branch1_ref) + ")"
            );
        }
        e.line(
            combiner_branch0_probability_local(i) + " = flat::balance_probability("
            + combiner_branch0_sample_albedo_local(i) + ", " + combiner_branch1_sample_albedo_local(i) + ")"
        );
        e.begin("if (" + combiner_active_local(i) + " <= 0.0)");
        e.line(combiner_summary_local(i, "roughness") + " = RoughnessInformation()");
        e.line(combiner_summary_local(i, "ior") + " = float3(0.0)");
        e.end();
    }
    if (!desc.combiners.empty())
        e.end();

    e.begin("// Calculate BSDF sampling probabilities.");
    emit_sample_probabilities(e, desc);
    e.end();

    e.begin("// Fold BSDF contribution scales into the persistent weights.");
    emit_final_bsdf_weights(e, desc);
    e.end();

    e.begin("// Assign root aggregate values.");
    e.line("root_albedo = " + ref_albedo(desc.root_ref));
    e.line("root_reflection = " + ref_reflection(desc.root_ref));
    e.line("root_transmission = " + ref_transmission(desc.root_ref));
    e.line("root_roughness = " + ref_roughness(desc.root_ref));
    e.line("root_ior_as_reflectance = " + ref_ior(desc.root_ref));
    e.end();
    e.end();

    e.begin(
        "public float3 collect_extra_bsdf_properties(inout ExtraBsdfProperties result, inout int reported_count, const "
        "MxExtraBsdfPropertiesContext ctx)"
    );
    for (size_t i = 0; i < desc.bsdfs.size(); ++i) {
        const int bsdf_index = int(i);
        e.line("{", false);
        e.line("    const float3 local_wi = " + bsdf_frame_field(bsdf_index) + ".to_local(ctx.wi_node)");
        e.line(
            "    const Frame leaf_frame_ws = mx_compose_child_frame_ws(ctx.node_frame_ws, "
            + bsdf_frame_field(bsdf_index) + ")"
        );
        e.line(
            "    mx_append_extra_bsdf_leaf(result, reported_count, " + desc.bsdfs[i].field_name
            + ", local_wi, ctx.eval_scale * " + bsdf_weight_field(bsdf_index)
            + ", leaf_frame_ws, ctx.diagnostic_frame_ws, " + std::to_string(i) + ")"
        );
        e.line("}", false);
    }
    e.line("return float3(0.0)");
    e.end();

    e.begin(
        "public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const "
        "BSDFContext bc)"
    );
    e.line("float3 result = float3(0.0)");
    for (size_t i = 0; i < desc.bsdfs.size(); ++i) {
        const int bsdf_index = int(i);
        e.line(
            "result += flat::eval_bsdf(" + desc.bsdfs[i].field_name + ", " + bsdf_weight_field(bsdf_index) + ", "
            + bsdf_frame_field(bsdf_index) + ", wi, wo, sg, bc)"
        );
    }
    e.line("return result");
    e.end();

    emit_bsdf_dispatch(e, desc);
    emit_cdf_sample_bsdf_selector(e, desc);
    emit_sample(e);
    emit_eval_pdf(e, desc);

    e.begin("public AlbedoContributions eval_albedo(const float3 wi)");
    e.line("// Return the closure-tree aggregate reconstructed above, not per-BSDF eval_albedo.");
    e.line("return flat::from_scattering_albedo(root_reflection, root_transmission)");
    e.end();

    e.begin("public RoughnessInformation eval_roughness(const float3 wi)");
    e.line("return root_roughness");
    e.end();

    e.line("public property float3 ior_as_reflectance");
    e.line("{");
    e.line("get { return root_ior_as_reflectance; }");
    e.line("}");
    e.end(";");

    return e.str();
}

} // namespace falcor::materialx::mx139
