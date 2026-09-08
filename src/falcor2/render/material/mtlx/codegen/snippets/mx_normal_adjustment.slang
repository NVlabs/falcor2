// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mtlx_snippets;

/// @brief Returns whether MaterialX surface-lobe normals should be repaired.
///
/// Keeping the policy behind this graph-local query lets configurations that always
/// enable or disable the repair return a constant without storing material hints.
bool mx_should_adjust_shading_normal()
{
    return bool(MaterialInstanceHints(hints) & MaterialInstanceHints::adjust_shading_normal);
}

/// @brief Converts and optionally repairs a MaterialX surface-lobe normal.
///
/// @param normal_ws BSDF MaterialX normal input in the generated shader's
/// surface vector space, which is world space for the active SlangPT bindings.
/// Normal/bump-map nodes have already expanded tangent-space texture values
/// through their normal/tangent inputs before reaching this point.
/// @return Normal in `si.shading_frame_ws` local coordinates. This is
/// the space stored in `mx_stack.bsdf_n[]` and used to build each BSDF
/// shading frame.
float3 adjust_normal(float3 normal_ws)
{
    const float3 candidate_normal_ls = normalize(si.shading_frame_ws.to_local(normal_ws));
    if (!mx_should_adjust_shading_normal())
        return candidate_normal_ls;

    const float3 view_direction_ls = normalize(si.shading_frame_ws.to_local(si.wi_ws));
    const float3 front_geometric_normal_ls = normalize(si.shading_frame_ws.to_local(si.front_normal_ws()));
    return adjust_shading_normal(candidate_normal_ls, front_geometric_normal_ls, view_direction_ls);
}
