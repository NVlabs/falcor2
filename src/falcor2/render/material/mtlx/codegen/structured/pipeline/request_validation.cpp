// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "request_validation.h"

#include "falcor2/core/error.h"

namespace falcor::mtlx::structured {

namespace {

void validate_optimize_graph_flags(const CodeGenDesc& desc)
{
    const uint32_t raw_flags = static_cast<uint32_t>(desc.optimize_graph_flags);
    const uint32_t known_flags = static_cast<uint32_t>(OptimizeGraphFlags::all);
    const uint32_t unknown_flags = raw_flags & ~known_flags;
    FALCOR_CHECK(
        unknown_flags == 0,
        "Unsupported Mtlx optimize graph flag bits 0x{:x}. Expected flags: closure_pruning, "
        "static_scatter_mode, fresnel_policy.",
        unknown_flags
    );
}

void validate_layering_mode(const CodeGenDesc& desc)
{
    switch (desc.layering_mode) {
    case LayeringMode::closure_tree:
    case LayeringMode::bsdf_mix:
        return;
    }
    FALCOR_THROW(
        "Unsupported Mtlx layering mode '{}'. Expected one of: closure_tree, bsdf_mix.",
        sgl::enum_to_string(desc.layering_mode)
    );
}

void validate_compensation_mode(const CodeGenDesc& desc)
{
    switch (desc.compensation_mode) {
    case CompensationMode::turquin_lut:
    case CompensationMode::turquin_analytic:
    case CompensationMode::no_compensation:
        return;
    }
    FALCOR_THROW(
        "Unsupported Mtlx compensation mode '{}'. Expected one of: turquin_lut, "
        "turquin_analytic, no_compensation.",
        sgl::enum_to_string(desc.compensation_mode)
    );
}

void validate_zeltner_sheen_coefficient_mode(const CodeGenDesc& desc)
{
    switch (desc.zeltner_sheen_coefficients) {
    case ZeltnerSheenCoefficientMode::analytic_fit:
    case ZeltnerSheenCoefficientMode::lut:
        return;
    }
    FALCOR_THROW(
        "Unsupported Mtlx Zeltner sheen coefficient mode '{}'. Expected one of: analytic_fit, lut.",
        sgl::enum_to_string(desc.zeltner_sheen_coefficients)
    );
}

} // namespace

void validate_request(const CodeGenDesc& desc)
{
    validate_optimize_graph_flags(desc);
    validate_layering_mode(desc);
    validate_compensation_mode(desc);
    validate_zeltner_sheen_coefficient_mode(desc);

    FALCOR_CHECK(!desc.positionfree_layering, "Structured MaterialX does not support position-free layering.");
    FALCOR_CHECK(
        desc.sigma_a.x == 0.0f && desc.sigma_a.y == 0.0f && desc.sigma_a.z == 0.0f && desc.sigma_s.x == 0.0f
            && desc.sigma_s.y == 0.0f && desc.sigma_s.z == 0.0f
            && std::holds_alternative<std::monostate>(desc.phase_function_params),
        "Structured MaterialX does not support legacy entry-point volume properties; encode volume and phase "
        "behavior in the MaterialX graph."
    );
}

} // namespace falcor::mtlx::structured
