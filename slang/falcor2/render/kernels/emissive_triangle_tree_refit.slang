// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

static const float NORMAL_SUM_EPSILON = 1e-20;

[shader("compute")]
[numthreads(128, 1, 1)]
void refit_leaves(
    uint thread_index: SV_DispatchThreadID,
    RWStructuredBuffer<PackedEmissiveTriangleTreeNode> nodes,
    RWStructuredBuffer<float4> weighted_normal_sums,
    StructuredBuffer<uint> node_indices,
    StructuredBuffer<EmissiveTriangleID> triangle_ids,
    uniform EmissiveGeometry emissive_geometry,
    uniform uint first_node_offset,
    uniform uint node_count
)
{
    if (thread_index >= node_count)
        return;

    const uint node_index = node_indices[first_node_offset + thread_index];
    const PackedEmissiveTriangleTreeNode old_node = nodes[node_index];
    const uint triangle_offset = old_node.triangle_offset();
    const uint triangle_count = old_node.triangle_count();

    float3 bounds_min = float3(float.MAX);
    float3 bounds_max = float3(-float.MAX);
    float flux = 0.0;
    float3 weighted_normal_sum = 0.0;
    for (uint i = 0; i < triangle_count; ++i) {
        const EmissiveTriangleID triangle_id = triangle_ids[triangle_offset + i];
        const EmissiveTriangle triangle = emissive_geometry.get_triangle(triangle_id);
        bounds_min = min(bounds_min, min(triangle.pos_ws[0], min(triangle.pos_ws[1], triangle.pos_ws[2])));
        bounds_max = max(bounds_max, max(triangle.pos_ws[0], max(triangle.pos_ws[1], triangle.pos_ws[2])));
        const float triangle_flux = emissive_geometry.get_triangle_flux(triangle_id);
        flux += triangle_flux;
        weighted_normal_sum += triangle.normal_ws * max(triangle_flux, NORMAL_SUM_EPSILON);
    }

    float3 cone_direction = 0.0;
    float cos_cone_angle = -1.0;
    const float cone_length = length(weighted_normal_sum);
    if (isfinite(cone_length) && cone_length > NORMAL_SUM_EPSILON) {
        cone_direction = weighted_normal_sum / cone_length;
        cos_cone_angle = 1.0;
        for (uint i = 0; i < triangle_count; ++i) {
            const EmissiveTriangleID triangle_id = triangle_ids[triangle_offset + i];
            const float3 normal = emissive_geometry.get_triangle(triangle_id).normal_ws;
            cos_cone_angle = min(cos_cone_angle, clamp(dot(cone_direction, normal), -1.0, 1.0));
        }
    }

    EmissiveTriangleTreeNode data;
    data.set_bounds(bounds_min, bounds_max);
    data.flux = flux;
    data.right_child = 0xffffffff;
    data.cone_direction = cone_direction;
    data.cos_cone_angle = cos_cone_angle;
    data.triangle_offset = triangle_offset;
    data.triangle_count = triangle_count;
    nodes[node_index] = detail::pack_emissive_triangle_tree_node(data);
    weighted_normal_sums[node_index] = float4(weighted_normal_sum, 0.0);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void refit_internal_nodes(
    uint thread_index: SV_DispatchThreadID,
    RWStructuredBuffer<PackedEmissiveTriangleTreeNode> nodes,
    RWStructuredBuffer<float4> weighted_normal_sums,
    StructuredBuffer<uint> node_indices,
    uniform uint first_node_offset,
    uniform uint node_count
)
{
    if (thread_index >= node_count)
        return;

    const uint node_index = node_indices[first_node_offset + thread_index];
    const PackedEmissiveTriangleTreeNode old_node = nodes[node_index];
    const uint left_child_index = node_index + 1;
    const uint right_child_index = old_node.right_child();
    const EmissiveTriangleTreeNode left = detail::unpack_emissive_triangle_tree_node(nodes[left_child_index]);
    const EmissiveTriangleTreeNode right = detail::unpack_emissive_triangle_tree_node(nodes[right_child_index]);

    const float3 weighted_normal_sum
        = weighted_normal_sums[left_child_index].xyz + weighted_normal_sums[right_child_index].xyz;
    float3 cone_direction = 0.0;
    float cos_cone_angle = -1.0;
    const float cone_length = length(weighted_normal_sum);
    if (isfinite(cone_length) && cone_length > NORMAL_SUM_EPSILON && left.cos_cone_angle > -1.0
        && right.cos_cone_angle > -1.0) {
        cone_direction = weighted_normal_sum / cone_length;
        const float left_cos_cone_angle = detail::widen_emissive_triangle_tree_cone_cos(
            left.cos_cone_angle,
            dot(cone_direction, left.cone_direction)
        );
        const float right_cos_cone_angle = detail::widen_emissive_triangle_tree_cone_cos(
            right.cos_cone_angle,
            dot(cone_direction, right.cone_direction)
        );
        cos_cone_angle = min(left_cos_cone_angle, right_cos_cone_angle);
        if (cos_cone_angle <= -1.0)
            cone_direction = 0.0;
    }

    EmissiveTriangleTreeNode data;
    data.set_bounds(
        min(left.get_bounds_min(), right.get_bounds_min()),
        max(left.get_bounds_max(), right.get_bounds_max())
    );
    data.flux = left.flux + right.flux;
    data.right_child = right_child_index;
    data.cone_direction = cone_direction;
    data.cos_cone_angle = cos_cone_angle;
    data.triangle_offset = 0;
    data.triangle_count = 0;
    nodes[node_index] = detail::pack_emissive_triangle_tree_node(data);
    weighted_normal_sums[node_index] = float4(weighted_normal_sum, 0.0);
}
