// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mx139;

import falcor2.render;

namespace mx139 {

#ifndef MX139_VALIDATE_BUMP_HEMISPHERE
#ifdef MX139_DIAGNOSTIC_MITSUBA_FRAME_GATE
#define MX139_VALIDATE_BUMP_HEMISPHERE MX139_DIAGNOSTIC_MITSUBA_FRAME_GATE
#else
#define MX139_VALIDATE_BUMP_HEMISPHERE 1
#endif
#endif

/// @brief Validates that the incoming direction stays in the same hemisphere after bump-frame remapping.
///
/// @param si Surface interaction containing the unperturbed outer shading frame.
/// @param wi_bump Incoming direction expressed in the perturbed BSDF/bump frame.
/// @return `true` when the outer shading frame and bumped BSDF frame agree on
/// whether the incoming direction is above or below the surface. This prevents
/// a perturbed normal from turning a reflection-side direction into a
/// refraction-side direction solely through frame mismatch.
bool validate_incoming_bump_hemisphere(const SurfaceInteraction si, const float3 wi_bump)
{
#if MX139_VALIDATE_BUMP_HEMISPHERE
    return dot(si.wi_ws, si.shading_frame_ws.normal) * wi_bump.z > 0.0;
#else
    return true;
#endif
}

/// @brief Validates that the outgoing direction stays in the same hemisphere after bump-frame remapping.
///
/// @param si Surface interaction containing the unperturbed outer shading frame.
/// @param wo_ws Outgoing direction in world space.
/// @param wo_bump Outgoing direction expressed in the perturbed BSDF/bump frame.
/// @return `true` when the outer shading frame and bumped BSDF frame agree on
/// whether the outgoing direction is above or below the surface. This prevents
/// a perturbed normal from turning a reflection-side direction into a
/// refraction-side direction solely through frame mismatch.
bool validate_outgoing_bump_hemisphere(const SurfaceInteraction si, const float3 wo_ws, const float3 wo_bump)
{
#if MX139_VALIDATE_BUMP_HEMISPHERE
    return dot(wo_ws, si.shading_frame_ws.normal) * wo_bump.z > 0.0;
#else
    return true;
#endif
}

/// @brief Validates an incoming/outgoing direction pair after bump-frame remapping.
///
/// @param si Surface interaction containing the unperturbed outer shading frame.
/// @param wi_bump Incoming direction expressed in the perturbed BSDF/bump frame.
/// @param wo_ws Outgoing direction in world space.
/// @param wo_bump Outgoing direction expressed in the perturbed BSDF/bump frame.
/// @return `true` when both directions remain in their corresponding outer-frame
/// hemispheres after bump-frame remapping.
bool validate_bump_hemisphere_pair(
    const SurfaceInteraction si,
    const float3 wi_bump,
    const float3 wo_ws,
    const float3 wo_bump
)
{
    return validate_incoming_bump_hemisphere(si, wi_bump) && validate_outgoing_bump_hemisphere(si, wo_ws, wo_bump);
}

} // namespace mx139
