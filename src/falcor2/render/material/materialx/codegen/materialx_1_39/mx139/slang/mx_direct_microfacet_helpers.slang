// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mx139_snippets;

void direct_prepare_bsdf(BSDF bsdf, float3 normal, float3 tangent)
{
    const int bsdf_index = bsdf.get_index();
    mx_stack.bsdf_n[bsdf_index] = adjust_normal(normal);
    mx_stack.bsdf_t[bsdf_index] = si.shading_frame_ws.to_local(tangent);
    orthonormalize(mx_stack.bsdf_n[bsdf_index], mx_stack.bsdf_t[bsdf_index]);
#ifdef MX139_STACK_HAS_BSDF_FRAMES
    mx_stack.bsdf_frames[bsdf_index] = Frame(mx_stack.bsdf_n[bsdf_index], mx_stack.bsdf_t[bsdf_index]);
#endif
}
