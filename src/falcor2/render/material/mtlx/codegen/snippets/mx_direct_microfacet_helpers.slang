// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mtlx_snippets;

[mutating]
void direct_prepare_bsdf(BSDF bsdf, float3 normal, float3 tangent)
{
    const int bsdf_index = bsdf.get_index();
    float3 N = adjust_normal(normal);
    float3 T = si.shading_frame_ws.to_local(tangent);
    orthonormalize(N, T);
#ifdef MTLX_SCALARIZED_GRAPH_STORAGE
    mx_stack.set_bsdf_frame(bsdf_index, N, T);
#else
    mx_stack.bsdf_n[bsdf_index] = N;
    mx_stack.bsdf_t[bsdf_index] = T;
#endif
#ifdef MTLX_STACK_HAS_BSDF_FRAMES
    mx_stack.bsdf_frames[bsdf_index] = Frame(mx_stack.bsdf_n[bsdf_index], mx_stack.bsdf_t[bsdf_index]);
#endif
}
