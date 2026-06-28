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
void compute_lss_aabbs(
    uint3 tid: SV_DispatchThreadID,
    StructuredBuffer<PackedLSSVertex> vertices,
    StructuredBuffer<uint> indices,
    RWStructuredBuffer<AABB> aabbs,
    uniform uint vertex_offset,
    uniform uint index_offset,
    uniform uint index_count
)
{
    if (tid.x >= index_count)
        return;

    // Successive indexing mode: segment i connects vertex index[i] and index[i]+1.
    uint segment_index = tid.x;
    uint idx = indices[index_offset + segment_index];
    LSSVertex v0 = detail::unpack_lss_vertex(vertices[vertex_offset + idx]);
    LSSVertex v1 = detail::unpack_lss_vertex(vertices[vertex_offset + idx + 1]);

    // Compute tight AABB enclosing both spheres.
    float3 min_corner = min(v0.position - v0.radius, v1.position - v1.radius);
    float3 max_corner = max(v0.position + v0.radius, v1.position + v1.radius);

    AABB aabb;
    aabb.min_corner = min_corner;
    aabb.max_corner = max_corner;
    aabbs[index_offset + segment_index] = aabb;
}
