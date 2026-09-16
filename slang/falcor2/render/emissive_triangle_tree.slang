// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Link-time selection of the leaf algorithm. Host samplers export the
/// selected value when composing a shader; standalone users get uniform mode.
extern static const uint EMISSIVE_TRIANGLE_TREE_LEAF_SAMPLING_MODE
    = uint(EmissiveTriangleTreeLeafSamplingMode::uniform);

/// Context-dependent sampler for active emissive triangles stored in a binary light hierarchy.
public struct EmissiveTriangleTree
{
    public StructuredBuffer<PackedEmissiveTriangleTreeNode> nodes;
    public StructuredBuffer<EmissiveTriangleID> triangle_ids;
    public StructuredBuffer<uint> active_triangle_leaf_indices;

    private static const uint INVALID_INDEX = 0xffffffff;
    private static const float INVALID_COS_CONE_ANGLE = -1.0;

    private float cos_sub_clamped(float sin_a, float cos_a, float sin_b, float cos_b)
    {
        if (cos_a > cos_b)
            return 1.0;
        return cos_a * cos_b + sin_a * sin_b;
    }

    private float sin_sub_clamped(float sin_a, float cos_a, float sin_b, float cos_b)
    {
        if (cos_a > cos_b)
            return 0.0;
        return sin_a * cos_b - cos_a * sin_b;
    }

    private float bound_cosine_term(
        const float3 position_ws,
        const float3 normal_ws,
        const float3 center,
        const float3 extent,
        out float spatial_cone_cos
    )
    {
        const float3 to_center = center - position_ws;
        const float distance_sqr = dot(to_center, to_center);
        const float radius_sqr = dot(extent, extent);
        if (distance_sqr <= radius_sqr) {
            spatial_cone_cos = -1.0;
            return 1.0;
        }

        const float sin_spatial_cone = sqrt(saturate(radius_sqr / distance_sqr));
        spatial_cone_cos = sqrt(max(0.0, 1.0 - sin_spatial_cone * sin_spatial_cone));
        const float3 direction = to_center / sqrt(distance_sqr);
        const float cos_direction = clamp(dot(normal_ws, direction), -1.0, 1.0);
        const float sin_direction = sqrt(max(0.0, 1.0 - cos_direction * cos_direction));
        return saturate(cos_sub_clamped(sin_direction, cos_direction, sin_spatial_cone, spatial_cone_cos));
    }

    private float compute_node_importance(const LightSampleContext ctx, const uint node_index)
    {
        const EmissiveTriangleTreeNode node = detail::unpack_emissive_triangle_tree_node(nodes[node_index]);
        const float3 to_center = node.center - ctx.pos_ws;
        float distance = length(to_center);

        const float flux = node.flux;
        if (flux <= 0.0)
            return 0.0;

        float spatial_cone_cos;
        const float cosine_bound
            = bound_cosine_term(ctx.pos_ws, ctx.normal_ws, node.center, node.extent, spatial_cone_cos);
        const float cosine_weight = dot(ctx.normal_ws, ctx.normal_ws) > 0.0 ? cosine_bound : 1.0;

        float orientation_weight = 1.0;
        if (node.cos_cone_angle != INVALID_COS_CONE_ANGLE && node.cos_cone_angle > 0.0 && distance > float.MIN) {
            const float3 direction_to_node = to_center / distance;
            const float cos_theta = clamp(dot(node.cone_direction, -direction_to_node), -1.0, 1.0);
            const float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
            const float sin_lighting_cone = sqrt(max(0.0, 1.0 - node.cos_cone_angle * node.cos_cone_angle));
            const float sin_spatial_cone = sqrt(max(0.0, 1.0 - spatial_cone_cos * spatial_cone_cos));
            const float cos_theta_0 = cos_sub_clamped(sin_theta, cos_theta, sin_lighting_cone, node.cos_cone_angle);
            const float sin_theta_0 = sin_sub_clamped(sin_theta, cos_theta, sin_lighting_cone, node.cos_cone_angle);
            orientation_weight
                = max(0.0, cos_sub_clamped(sin_theta_0, cos_theta_0, sin_spatial_cone, spatial_cone_cos));
        }

        const float half_radius = max(node.extent.x, max(node.extent.y, node.extent.z));
        distance = max(max(half_radius, distance), 1e-10);
        return flux * cosine_weight * orientation_weight / (distance * distance);
    }

    private float compute_triangle_importance(const LightSampleContext ctx, const EmissiveTriangleID triangle_id)
    {
        const EmissiveTriangle triangle = g_scene.emissive_geometry.get_triangle(triangle_id);
        if (dot(ctx.pos_ws - triangle.pos_ws[0], triangle.normal_ws) <= 0.0)
            return 0.0;

        const float distance_sqr = max(
            1e-10,
            point_triangle_distance_squared(ctx.pos_ws, triangle.pos_ws[0], triangle.pos_ws[1], triangle.pos_ws[2])
        );
        float cosine_weight = 1.0;
        if (dot(ctx.normal_ws, ctx.normal_ws) > 0.0) {
            cosine_weight = 0.0;
            for (uint i = 0; i < 3; ++i) {
                const float3 direction = triangle.pos_ws[i] - ctx.pos_ws;
                const float direction_length_sqr = dot(direction, direction);
                if (direction_length_sqr > float.MIN) {
                    cosine_weight = max(cosine_weight, dot(ctx.normal_ws, direction * rsqrt(direction_length_sqr)));
                }
            }
        }
        return g_scene.emissive_geometry.get_triangle_flux(triangle_id) * saturate(cosine_weight) / distance_sqr;
    }

    private float point_segment_distance_squared(const float3 point, const float3 begin, const float3 end)
    {
        const float3 edge = end - begin;
        const float edge_length_sqr = dot(edge, edge);
        const float t = edge_length_sqr > float.MIN ? saturate(dot(point - begin, edge) / edge_length_sqr) : 0.0;
        const float3 offset = point - (begin + t * edge);
        return dot(offset, offset);
    }

    private float point_triangle_distance_squared(
        const float3 point,
        const float3 vertex0,
        const float3 vertex1,
        const float3 vertex2
    )
    {
        const float3 edge0 = vertex1 - vertex0;
        const float3 edge1 = vertex2 - vertex0;
        const float3 point_offset = point - vertex0;
        const float d00 = dot(edge0, edge0);
        const float d01 = dot(edge0, edge1);
        const float d11 = dot(edge1, edge1);
        const float d20 = dot(point_offset, edge0);
        const float d21 = dot(point_offset, edge1);
        const float denominator = d00 * d11 - d01 * d01;
        const float denominator_scale = max(d00 * d11, float.MIN);

        if (denominator > 1e-12 * denominator_scale) {
            const float barycentric1 = (d11 * d20 - d01 * d21) / denominator;
            const float barycentric2 = (d00 * d21 - d01 * d20) / denominator;
            const float barycentric0 = 1.0 - barycentric1 - barycentric2;
            if (barycentric0 >= 0.0 && barycentric1 >= 0.0 && barycentric2 >= 0.0) {
                const float3 normal = cross(edge0, edge1);
                const float plane_offset = dot(point_offset, normal);
                return plane_offset * plane_offset / denominator;
            }
        }

        return min(
            point_segment_distance_squared(point, vertex0, vertex1),
            min(point_segment_distance_squared(point, vertex1, vertex2),
                point_segment_distance_squared(point, vertex2, vertex0))
        );
    }

    private bool traverse(const LightSampleContext ctx, inout float u, out float pdf, out uint node_index)
    {
        pdf = 1.0;
        node_index = 0;
        while (!nodes[node_index].is_leaf()) {
            const uint left_index = node_index + 1;
            const uint right_index = nodes[node_index].right_child();
            const float left_importance = compute_node_importance(ctx, left_index);
            const float right_importance = compute_node_importance(ctx, right_index);
            const float total_importance = left_importance + right_importance;
            if (total_importance <= 0.0)
                return false;

            const float left_probability = left_importance / total_importance;
            if (u < left_probability) {
                u /= max(left_probability, float.MIN);
                pdf *= left_probability;
                node_index = left_index;
            } else {
                const float right_probability = 1.0 - left_probability;
                u = (u - left_probability) / max(right_probability, float.MIN);
                pdf *= right_probability;
                node_index = right_index;
            }
        }
        return true;
    }

    private bool sample_leaf(
        const LightSampleContext ctx,
        const uint node_index,
        const float u,
        out float pdf,
        out EmissiveTriangleID triangle_id
    )
    {
        pdf = 0.0;
        triangle_id = EmissiveTriangleID::invalid;
        const EmissiveTriangleTreeNode node = detail::unpack_emissive_triangle_tree_node(nodes[node_index]);
        if (node.triangle_count == 1) {
            triangle_id = triangle_ids[node.triangle_offset];
            pdf = 1.0;
            return true;
        }
        if (EMISSIVE_TRIANGLE_TREE_LEAF_SAMPLING_MODE == uint(EmissiveTriangleTreeLeafSamplingMode::uniform)) {
            const uint index = min(uint(u * node.triangle_count), node.triangle_count - 1);
            triangle_id = triangle_ids[node.triangle_offset + index];
            pdf = 1.0 / float(node.triangle_count);
            return true;
        }

        float total_importance = 0.0;
        for (uint i = 0; i < node.triangle_count; ++i)
            total_importance += compute_triangle_importance(ctx, triangle_ids[node.triangle_offset + i]);
        if (total_importance <= 0.0)
            return false;

        const float target = u * total_importance;
        float cdf = 0.0;
        uint selected_index = node.triangle_count - 1;
        float selected_importance = 0.0;
        for (uint i = 0; i < node.triangle_count; ++i) {
            const float importance = compute_triangle_importance(ctx, triangle_ids[node.triangle_offset + i]);
            cdf += importance;
            if (target < cdf) {
                selected_index = i;
                selected_importance = importance;
                break;
            }
        }
        triangle_id = triangle_ids[node.triangle_offset + selected_index];
        if (selected_importance == 0.0)
            selected_importance = compute_triangle_importance(ctx, triangle_id);
        pdf = selected_importance / total_importance;
        return pdf > 0.0;
    }

    private float
    eval_leaf_pdf(const LightSampleContext ctx, const uint node_index, const EmissiveTriangleID triangle_id)
    {
        const EmissiveTriangleTreeNode node = detail::unpack_emissive_triangle_tree_node(nodes[node_index]);
        if (node.triangle_count == 1)
            return 1.0;
        if (EMISSIVE_TRIANGLE_TREE_LEAF_SAMPLING_MODE == uint(EmissiveTriangleTreeLeafSamplingMode::uniform))
            return 1.0 / float(node.triangle_count);

        float selected_importance = 0.0;
        float total_importance = 0.0;
        for (uint i = 0; i < node.triangle_count; ++i) {
            const EmissiveTriangleID local_id = triangle_ids[node.triangle_offset + i];
            const float importance = compute_triangle_importance(ctx, local_id);
            if (local_id == triangle_id)
                selected_importance = importance;
            total_importance += importance;
        }
        return total_importance > 0.0 ? selected_importance / total_importance : 0.0;
    }

    private float eval_traversal_pdf(const LightSampleContext ctx, const uint target_leaf_index)
    {
        float pdf = 1.0;
        uint node_index = 0;
        while (node_index != target_leaf_index) {
            if (nodes[node_index].is_leaf())
                return 0.0;

            const uint left_index = node_index + 1;
            const uint right_index = nodes[node_index].right_child();
            const float left_importance = compute_node_importance(ctx, left_index);
            const float right_importance = compute_node_importance(ctx, right_index);
            const float total_importance = left_importance + right_importance;
            if (total_importance <= 0.0)
                return 0.0;

            const float left_probability = left_importance / total_importance;
            if (target_leaf_index < right_index) {
                pdf *= left_probability;
                node_index = left_index;
            } else {
                pdf *= 1.0 - left_probability;
                node_index = right_index;
            }
        }
        return nodes[node_index].is_leaf() ? pdf : 0.0;
    }

    public bool
    sample(const LightSampleContext ctx, inout float u, out float selection_pdf, out EmissiveTriangleID triangle_id)
    {
        selection_pdf = 0.0;
        triangle_id = EmissiveTriangleID::invalid;
        if (g_scene.emissive_geometry.active_triangle_count == 0)
            return false;

        float traversal_pdf;
        uint leaf_index;
        if (!traverse(ctx, u, traversal_pdf, leaf_index))
            return false;

        float leaf_pdf;
        if (!sample_leaf(ctx, leaf_index, u, leaf_pdf, triangle_id))
            return false;

        selection_pdf = traversal_pdf * leaf_pdf;
        return selection_pdf > 0.0;
    }

    public float eval_pdf(const LightSampleContext ctx, const EmissiveTriangleID triangle_id)
    {
        if (g_scene.emissive_geometry.active_triangle_count == 0 || triangle_id == EmissiveTriangleID::invalid
            || uint(triangle_id) >= g_scene.emissive_geometry.triangle_count)
            return 0.0;

        const ActiveEmissiveTriangleID active_id
            = g_scene.emissive_geometry.triangle_id_to_active_triangle_id[uint(triangle_id)];
        if (active_id == ActiveEmissiveTriangleID::invalid)
            return 0.0;

        const uint leaf_index = active_triangle_leaf_indices[uint(active_id)];
        return eval_traversal_pdf(ctx, leaf_index) * eval_leaf_pdf(ctx, leaf_index, triangle_id);
    }
}
