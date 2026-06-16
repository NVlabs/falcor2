// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"
#include <MaterialXGenShader/Shader.h>

#include <string>
#include <string_view>

namespace falcor {
namespace materialx {
namespace mx139 {

class StaticInputQuery;

// Resolve the effective set of optimization flags after codegen constraints
// such as editable graphs.
OptimizeGraphFlags effective_optimize_graph_flags(const CodeGenDesc& desc);

// Identifies the MaterialX microfacet BSDF implementation family that may be
// specialized by Mx139 policy code. These values describe shader node
// implementations only; selecting a kind does not mutate the graph.
enum class MxMicrofacetKind {
    none,
    dielectric,
    conductor,
    generalized_schlick,
};

// The Fresnel and scatter selections are the static policy decisions encoded
// into generated Mx139 BSDF type names and BSDF constructor calls.
enum class MxFresnelSelection {
    standard,
    color82,
    airy,
};

enum class MxScatterSelection {
    runtime,
    static_r,
    static_t,
    static_rt,
};

// Aggregate policy selected for one Mx139 microfacet shader node. It is derived
// from CodeGenDesc plus statically-known shader inputs and constants; it carries
// no graph ownership and performs no graph mutation.
struct MxMicrofacetSelection {
    MxMicrofacetKind kind = MxMicrofacetKind::none;
    MxFresnelSelection fresnel = MxFresnelSelection::standard;
    MxScatterSelection scatter = MxScatterSelection::runtime;
};

MxMicrofacetKind microfacet_kind(const std::string& implementation);

// Returns the MaterialX scatter enum spelling represented by a static
// selection. Runtime selection preserves the MaterialX default RT spelling.
std::string scatter_selection_mode_string(MxScatterSelection selection);

// Selects the Mx139 microfacet policy for a shader node. The optional constants
// block is still used by scatter-mode handling. Fresnel simplification asks the
// optional StaticInputQuery for positive exact-value proofs instead of reading
// implementation defaults directly.
MxMicrofacetSelection select_mx139_microfacet(
    const CodeGenDesc& desc,
    const std::string& implementation,
    const MaterialX::ShaderNode* node,
    const MaterialX::VariableBlock* constants = nullptr,
    const StaticInputQuery* static_input_query = nullptr
);

// Converts a selected policy into the generated Mx139 BSDF type name used by
// layering analysis and BSDF emitters.
std::string microfacet_bsdf_type(const CodeGenDesc& desc, MxMicrofacetSelection selection);

// Helpers for code paths that need the lobe/transmission meaning of MaterialX
// scatter-mode values without re-running the full microfacet selection.
bool scatter_mode_includes_transmission(std::string_view scatter_mode);
std::string glossy_scatter_lobe_types(std::string_view scatter_mode);
bool is_scatter_mode_bsdf(const std::string& implementation);

} // namespace mx139
} // namespace materialx
} // namespace falcor
