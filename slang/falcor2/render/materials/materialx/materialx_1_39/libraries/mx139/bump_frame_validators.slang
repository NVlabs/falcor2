// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Bump-frame hemisphere validation helpers for MX139 local frame adapters.

implementing mx139;

namespace mx139 {

#ifndef MX139_VALIDATE_BUMP_HEMISPHERE
#ifdef MX139_DIAGNOSTIC_MITSUBA_FRAME_GATE
#define MX139_VALIDATE_BUMP_HEMISPHERE MX139_DIAGNOSTIC_MITSUBA_FRAME_GATE
#else
#define MX139_VALIDATE_BUMP_HEMISPHERE 1
#endif
#endif

/// @brief Validates that bump-frame remapping preserves the direction hemisphere.
///
/// @param outer_w Direction in the unperturbed outer shading frame.
/// @param bump_w The same direction expressed in the perturbed BSDF/bump frame.
/// @return `true` when both frames classify the direction as lying on the same
/// side of the surface. This rejects cases where a bumped BSDF frame would make
/// a reflection direction look like refraction, or a refraction direction look
/// like reflection, solely because the bump normal diverged from the outer
/// shading normal.
public bool validate_bump_frame_hemisphere(const float3 outer_w, const float3 bump_w)
{
#if MX139_VALIDATE_BUMP_HEMISPHERE
    return outer_w.z * bump_w.z > 0.0;
#else
    return true;
#endif
}

public bool validate_bump_frame_hemisphere_pair(
    const float3 outer_wi,
    const float3 bump_wi,
    const float3 outer_wo,
    const float3 bump_wo
)
{
    return validate_bump_frame_hemisphere(outer_wi, bump_wi) && validate_bump_frame_hemisphere(outer_wo, bump_wo);
}

} // namespace mx139
