// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

struct AABB
{
    float3 min_corner;
    float3 max_corner;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_lss_aabbs(uint3 tid: SV_DispatchThreadID, StructuredBuffer<LSSAABBWork> work, uniform uint2 thread_offset)
{
    const LSSAABBWork record = work[thread_offset.y + tid.y];
    const uint segment_index = thread_offset.x + tid.x;
    if (segment_index >= record.index_count)
        return;
    const uint vertex_offset = record.vertex_offset;
    const uint index_offset = record.index_offset;
    let vertices = record.vertex_buffer.as<PackedLSSVertex>();
    let indices = record.index_buffer.as<uint>();

    // Successive indexing mode: segment i connects vertex index[i] and index[i]+1.
    uint idx = indices[index_offset + segment_index];
    LSSVertex v0 = detail::unpack_lss_vertex(vertices[vertex_offset + idx]);
    LSSVertex v1 = detail::unpack_lss_vertex(vertices[vertex_offset + idx + 1]);

    // Compute tight AABB enclosing both spheres.
    float3 min_corner = min(v0.position - v0.radius, v1.position - v1.radius);
    float3 max_corner = max(v0.position + v0.radius, v1.position + v1.radius);

    AABB aabb;
    aabb.min_corner = min_corner;
    aabb.max_corner = max_corner;
    // The float2 view gives CUDA three 64-bit stores instead of six scalar AABB stores,
    // without changing the 24-byte AABB layout. Explicit alignment alone does not achieve this.
    // The host checks 8-byte base alignment; the 24-byte element stride preserves it.
    record.aabb_buffer.as<float2[3]>()[index_offset + segment_index] = reinterpret<float2[3]>(aabb);
}
