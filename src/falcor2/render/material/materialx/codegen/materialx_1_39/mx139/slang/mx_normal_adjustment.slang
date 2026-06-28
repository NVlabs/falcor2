// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mx139_snippets;

#ifndef MX139_DIAGNOSTIC_USE_MX_BEND_NORMAL
#define MX139_DIAGNOSTIC_USE_MX_BEND_NORMAL 0
#endif

#ifndef MX139_VALIDATE_BUMP_HEMISPHERE
#ifdef MX139_DIAGNOSTIC_MITSUBA_FRAME_GATE
#define MX139_VALIDATE_BUMP_HEMISPHERE MX139_DIAGNOSTIC_MITSUBA_FRAME_GATE
#else
#define MX139_VALIDATE_BUMP_HEMISPHERE 1
#endif
#endif

float3 bend_normal(float3 normal_ws)
{
    // Mirrors MaterialX 1.38's mxBendNormal(Ng, Nmapped, viewDir) heuristic.
    Frame sf = Frame(normalize(normal_ws));
    adjust_shading_normal(sf, si.front_normal_ws(), si.wi_ws);
    return si.shading_frame_ws.to_local(sf.normal);
}

float3 bsdl_visible_normal(float3 normal_ws)
{
    // BSDL repairs each lobe frame with BsdfGlobals::visible_normal(wo, Ngf, N).
    // Keep this policy isolated so the 1.38-style bend-normal diagnostic can be
    // switched without hiding policy branches inside individual BSDF helpers.
    const float3 wo = normalize(si.shading_frame_ws.to_local(si.wi_ws));
    const float3 ng = normalize(si.shading_frame_ws.to_local(si.normal_ws));
    const float3 n = normalize(si.shading_frame_ws.to_local(normal_ws));
    if (dot(wo, n) > 0.0)
        return n;

    var v = cross(wo, n);
    if (mx_hmax(abs(v)) < 1.0e-4) {
        v = cross(wo, ng);
        if (mx_hmax(abs(v)) < 1.0e-4) {
            const float s = select(wo.z < 0.0, -1.0, 1.0);
            const float a = -1.0 / (s + wo.z);
            v = float3(wo.x * wo.y * a, s + wo.y * wo.y * a, -wo.y);
        } else {
            v = normalize(v);
        }
    } else {
        v = normalize(v);
    }

    return normalize(cross(v, wo) + 1.0e-4 * wo);
}

/// @brief Applies the active BSDF-normal visibility repair policy.
///
/// @param normal_ws BSDF MaterialX normal input in the generated shader's
/// surface vector space, which is world space for the active SlangPT bindings.
/// Normal/bump-map nodes have already expanded tangent-space texture values
/// through their normal/tangent inputs before reaching this point.
/// @return Repaired normal in `si.shading_frame_ws` local coordinates. This is
/// the space stored in `mx_stack.bsdf_n[]` and used to build each BSDF
/// shading frame.
float3 adjust_normal(float3 normal_ws)
{
#if MX139_DIAGNOSTIC_USE_MX_BEND_NORMAL
    return bend_normal(normal_ws);
#else
    return bsdl_visible_normal(normal_ws);
#endif
}

/// @brief Validates that bump-frame remapping preserves the direction hemisphere.
///
/// @param outer_w Direction in the unperturbed outer shading frame.
/// @param bump_w The same direction expressed in the perturbed BSDF/bump frame.
/// @return `true` when both frames classify the direction as lying on the same
/// side of the surface. This rejects cases where a bumped BSDF frame would make
/// a reflection direction look like refraction, or a refraction direction look
/// like reflection, solely because the bump normal diverged from the outer
/// shading normal.
bool validate_bump_hemisphere(float3 outer_w, float3 bump_w)
{
    return outer_w.z * bump_w.z > 0.0;
}

/// @brief Validates the incoming direction against the perturbed bump frame.
///
/// @param wi_bump Incoming direction expressed in the perturbed BSDF/bump frame.
/// @return `true` when the active bump-hemisphere policy accepts the incoming
/// direction. The check compares the incoming direction in the outer shading
/// frame to the same direction in the bumped frame, rejecting cases where the
/// bump would convert reflection-side travel into refraction-side travel.
bool validate_incoming_bump_hemisphere(float3 wi_bump)
{
#if MX139_VALIDATE_BUMP_HEMISPHERE
    return validate_bump_hemisphere(si.shading_frame_ws.to_local(si.wi_ws), wi_bump);
#else
    return true;
#endif
}

/// @brief Validates the outgoing direction against the perturbed bump frame.
///
/// @param wo_ws Outgoing direction in world space.
/// @param wo_bump Outgoing direction expressed in the perturbed BSDF/bump frame.
/// @return `true` when the active bump-hemisphere policy accepts the outgoing
/// direction. The check compares the outgoing direction in the outer shading
/// frame to the same direction in the bumped frame, rejecting cases where the
/// bump would convert reflection-side travel into refraction-side travel.
bool validate_outgoing_bump_hemisphere(float3 wo_ws, float3 wo_bump)
{
#if MX139_VALIDATE_BUMP_HEMISPHERE
    return validate_bump_hemisphere(si.shading_frame_ws.to_local(wo_ws), wo_bump);
#else
    return true;
#endif
}

/// @brief Validates an incoming/outgoing direction pair against the perturbed bump frame.
///
/// @param wi_bump Incoming direction expressed in the perturbed BSDF/bump frame.
/// @param wo_ws Outgoing direction in world space.
/// @param wo_bump Outgoing direction expressed in the perturbed BSDF/bump frame.
/// @return `true` when both directions remain in their corresponding outer-frame
/// hemispheres after bump-frame remapping.
bool validate_bump_hemisphere_pair(float3 wi_bump, float3 wo_ws, float3 wo_bump)
{
    return validate_incoming_bump_hemisphere(wi_bump) && validate_outgoing_bump_hemisphere(wo_ws, wo_bump);
}
