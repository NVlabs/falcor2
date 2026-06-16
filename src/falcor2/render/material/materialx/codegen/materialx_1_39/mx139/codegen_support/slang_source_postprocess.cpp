// SPDX-License-Identifier: Apache-2.0

#include "slang_source_postprocess.h"

#include "string_utils.h"

#include "falcor2/core/error.h"

#include <regex>
#include <sstream>
#include <string_view>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace codegen_support {

void strip_slang_snippet_file_header(std::string& text)
{
    constexpr std::string_view k_spdx_prefix = "// SPDX-License-Identifier:";
    if (!text.starts_with(k_spdx_prefix))
        return;

    const size_t line_end = text.find('\n');
    text.erase(0, line_end == std::string::npos ? text.size() : line_end + 1);
    if (text.starts_with('\n'))
        text.erase(0, 1);
}

void strip_slang_snippet_module_marker(std::string& text)
{
    constexpr std::string_view k_module_marker = "implementing mx139_snippets;";
    if (!text.starts_with(k_module_marker))
        return;

    const size_t line_end = text.find('\n');
    text.erase(0, line_end == std::string::npos ? text.size() : line_end + 1);
    if (text.starts_with('\n'))
        text.erase(0, 1);
}

void replace_fresnel_factory_calls(std::string& source, const CodeGenDesc& desc)
{
    FALCOR_UNUSED(desc);

    replace_all(source, "create_mx_dielectric(", "create_mx_dielectric_airy(");
    replace_all(source, "create_mx_conductor(", "create_mx_conductor_airy(");
    replace_all(source, "create_mx_generalized_schlick(", "create_mx_generalized_schlick_airy(");
}

void replace_compensation_factory_calls(std::string& source, const CodeGenDesc& desc)
{
    if (desc.compensation_mode == CompensationMode::turquin_lut)
        return;

    std::string suffix;
    if (desc.compensation_mode == CompensationMode::turquin_analytic)
        suffix = "_comp_turquin_analytic";
    else if (desc.compensation_mode == CompensationMode::no_compensation)
        suffix = "_comp_no_compensation";
    else
        FALCOR_UNREACHABLE();
    replace_all(
        source,
        "create_mx_generalized_schlick_color82(",
        "create_mx_generalized_schlick_color82" + suffix + "("
    );
    replace_all(source, "create_mx_generalized_schlick_airy(", "create_mx_generalized_schlick_airy" + suffix + "(");
    replace_all(source, "create_mx_generalized_schlick(", "create_mx_generalized_schlick" + suffix + "(");
    replace_all(source, "create_mx_dielectric_airy(", "create_mx_dielectric_airy" + suffix + "(");
    replace_all(source, "create_mx_dielectric(", "create_mx_dielectric" + suffix + "(");
    replace_all(source, "create_mx_conductor_airy(", "create_mx_conductor_airy" + suffix + "(");
    replace_all(source, "create_mx_conductor(", "create_mx_conductor" + suffix + "(");
}

void replace_hw_geom_tokens(std::string& source)
{
    replace_all(source, "$vd.$positionWorld", "si.position_ws");
    replace_all(source, "$vd.$normalWorld", "si.shading_frame_ws.normal");
    replace_all(source, "$vd.$tangentWorld", "si.shading_frame_ws.tangent");
    replace_all(source, "$vd.$bitangentWorld", "si.shading_frame_ws.bitangent");
    replace_all(source, "$vd.$texcoord_0", "si.uv");
    replace_all(source, "$vd.$texcoord", "si.uv");
}

void mark_generated_void_methods_mutating(std::string& source)
{
    std::regex keyword_void("\\bvoid\\b");
    std::ostringstream oss;
    std::regex_replace(
        std::ostreambuf_iterator<char>(oss),
        source.begin(),
        source.end(),
        keyword_void,
        "[mutating] void"
    );
    source = oss.str();

    // The GLSL noise library helper mutates only its inout hash arguments. If it is marked
    // as a mutating graph method, non-void hash methods cannot call it on an immutable graph.
    replace_all(source, "[mutating] void mx_bjmix(", "void mx_bjmix(");

    // Advanced Mx139 layering implements the 1.38 generic material instance weight
    // calculator interface. Those methods only write through out parameters and must
    // match the imported interface exactly.
    replace_all(source, "public [mutating] void calculate_weights<", "public void calculate_weights<");
    replace_all(source, "[mutating] void calculate_weights<", "public void calculate_weights<");
    replace_all(source, "public [mutating] void calculate_plain_weights<", "public void calculate_plain_weights<");
    replace_all(source, "[mutating] void calculate_plain_weights<", "public void calculate_plain_weights<");
}

void replace_glsl_compatibility_tokens(std::string& source)
{
    replace_all(source, "M_PI_INV", "(1.0 / float.PI)");
    replace_all(source, "M_PI", "float.PI");
}

} // namespace codegen_support
} // namespace mx139
} // namespace materialx
} // namespace falcor
