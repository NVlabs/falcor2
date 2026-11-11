// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

[shader("compute")]
[numthreads(64, 1, 1)]
void animate_mesh(
    uint3 tid: SV_DispatchThreadID,
    StructuredBuffer<GeometryAnimationWork> work,
    uniform uint2 thread_offset
)
{
    let record = work[thread_offset.y + tid.y];
    uint vertex = thread_offset.x + tid.x;
    if (vertex >= record.vertex_count)
        return;
    PackedTriangleVertex current = record.current.as<PackedTriangleVertex>()[vertex];
    if (record.history_update != GeometryHistoryUpdate::initialize)
        record.previous.as<PackedTriangleVertex>()[vertex] = current;
    if (record.history_update == GeometryHistoryUpdate::catch_up)
        return;

    let a = detail::unpack_triangle_vertex(record.source0.as<PackedTriangleVertex>()[vertex]);
    let b = detail::unpack_triangle_vertex(record.source1.as<PackedTriangleVertex>()[vertex]);
    current.position = lerp(a.position, b.position, record.source1_weight);
    detail::pack_vertex_normal(
        current.normal_and_tangent,
        normalize_or(lerp(a.normal, b.normal, record.source1_weight), a.normal)
    );
    detail::pack_vertex_tangent(
        current.normal_and_tangent,
        normalize_or(lerp(a.tangent, b.tangent, record.source1_weight), a.tangent)
    );
    record.current.as<PackedTriangleVertex>()[vertex] = current;
    if (record.history_update == GeometryHistoryUpdate::initialize)
        record.previous.as<PackedTriangleVertex>()[vertex] = current;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void animate_lss(
    uint3 tid: SV_DispatchThreadID,
    StructuredBuffer<GeometryAnimationWork> work,
    uniform uint2 thread_offset
)
{
    let record = work[thread_offset.y + tid.y];
    uint vertex = thread_offset.x + tid.x;
    if (vertex >= record.vertex_count)
        return;
    PackedLSSVertex current = record.current.as<PackedLSSVertex>()[vertex];
    if (record.history_update != GeometryHistoryUpdate::initialize)
        record.previous.as<PackedLSSVertex>()[vertex] = current;
    if (record.history_update == GeometryHistoryUpdate::catch_up)
        return;

    let a = record.source0.as<PackedLSSVertex>()[vertex];
    let b = record.source1.as<PackedLSSVertex>()[vertex];
    current.position = lerp(a.position, b.position, record.source1_weight);
    current.radius = lerp(a.radius, b.radius, record.source1_weight);
    record.current.as<PackedLSSVertex>()[vertex] = current;
    if (record.history_update == GeometryHistoryUpdate::initialize)
        record.previous.as<PackedLSSVertex>()[vertex] = current;
}
