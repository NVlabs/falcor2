// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../codegen_types.h"
#include "bsdf_support.h"
#include <MaterialXGenShader/Shader.h>

#include <compare>
#include <string>
#include <string_view>

namespace falcor {
namespace mtlx::structured {

class StaticInputQuery;

// Identifies the MaterialX microfacet BSDF implementation family that may be
// specialized by Mtlx policy code. These values describe shader node
// implementations only; selecting a kind does not mutate the graph.
enum class MxMicrofacetKind {
    none,
    dielectric,
    conductor,
    generalized_schlick,
};

// Fresnel, scatter, and compensation are the static policy decisions encoded
// into generated Mtlx BSDF type names and BSDF constructor calls.
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

// Aggregate policy selected for one Mtlx microfacet shader node. It is derived
// from CodeGenDesc plus statically-known shader inputs and constants; it carries
// no graph ownership and performs no graph mutation.
struct MxMicrofacetSelection {
    MxMicrofacetKind kind = MxMicrofacetKind::none;
    MxFresnelSelection fresnel = MxFresnelSelection::standard;
    MxScatterSelection scatter = MxScatterSelection::runtime;
    CompensationMode compensation = CompensationMode::turquin_analytic;
    auto operator<=>(const MxMicrofacetSelection&) const = default;
};

/// Aggregate coefficient policy selected for one MaterialX sheen node.
struct MxSheenSelection {
    ZeltnerSheenCoefficientMode coefficients = ZeltnerSheenCoefficientMode::analytic_fit;
    auto operator<=>(const MxSheenSelection&) const = default;
};

MxMicrofacetKind microfacet_kind(BsdfKind kind);

// Returns the MaterialX scatter enum spelling represented by a static
// selection. Runtime selection preserves the MaterialX default RT spelling.
std::string scatter_selection_mode_string(MxScatterSelection selection);

// Selects the Mtlx microfacet policy for a shader node. The optional constants
// block is still used by scatter-mode handling. Fresnel simplification asks the
// optional StaticInputQuery for positive exact-value proofs instead of reading
// implementation defaults directly.
MxMicrofacetSelection select_microfacet(
    const CodeGenDesc& desc,
    BsdfKind kind,
    const MaterialX::ShaderNode* node,
    const MaterialX::VariableBlock* constants = nullptr,
    const StaticInputQuery* static_input_query = nullptr
);

// Converts a selected policy into the generated Mtlx BSDF type name used by
// layering analysis and BSDF emitters.
std::string microfacet_bsdf_type(MxMicrofacetSelection selection);

/// Return the least-specialized policy that correctly represents both uses of
/// one shared MaterialX implementation node.
MxMicrofacetSelection common_microfacet_selection(MxMicrofacetSelection a, MxMicrofacetSelection b);

MxSheenSelection select_sheen(const CodeGenDesc& desc);
std::string sheen_bsdf_type(MxSheenSelection selection);

// Helpers for code paths that need the lobe/transmission meaning of MaterialX
// scatter-mode values without re-running the full microfacet selection.
bool scatter_mode_includes_reflection(std::string_view scatter_mode);
bool scatter_mode_includes_transmission(std::string_view scatter_mode);
bool is_scatter_mode_bsdf(BsdfKind kind);

} // namespace mtlx::structured
} // namespace falcor
