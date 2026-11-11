// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

public ParameterBlock<UniformLightSampler> uniform_light_sampler;
public ParameterBlock<PowerLightSampler> power_light_sampler;
public ParameterBlock<HierarchicalLightSampler> hierarchical_light_sampler;

public float environment_direction_pdf(float3 direction)
{
    LightSampleContext ctx = {};
    return uniform_light_sampler.eval_environment_pdf(ctx, direction);
}

public float component_light_power(uint light_id)
{
    return g_scene.light_system.get_light(LightID(light_id)).power_estimate();
}

public float uniform_source_pdf(uint emitter_type, uint source_index)
{
    return uniform_light_sampler.eval_source_selection_pdf();
}

public float power_source_pdf(uint emitter_type, uint source_index)
{
    return power_light_sampler.eval_source_selection_pdf(EmitterType(emitter_type), source_index);
}

public float uniform_emissive_pdf(uint triangle_id)
{
    LightSampleContext ctx = {};
    return uniform_light_sampler
        .eval_emissive_pdf(ctx, EmissiveTriangleID(triangle_id), float3(0.0), float3(0.0, 0.0, 1.0));
}

public float power_emissive_pdf(uint triangle_id)
{
    LightSampleContext ctx = {};
    return power_light_sampler
        .eval_emissive_pdf(ctx, EmissiveTriangleID(triangle_id), float3(0.0), float3(0.0, 0.0, 1.0));
}

public float
uniform_area_light_pdf(uint light_id, float3 receiver_pos_ws, float3 emitter_pos_ws, float3 emitter_normal_ws)
{
    LightSampleContext ctx = {};
    ctx.pos_ws = receiver_pos_ws;
    return uniform_light_sampler.eval_area_light_pdf(ctx, LightID(light_id), emitter_pos_ws, emitter_normal_ws);
}

public float
power_area_light_pdf(uint light_id, float3 receiver_pos_ws, float3 emitter_pos_ws, float3 emitter_normal_ws)
{
    LightSampleContext ctx = {};
    ctx.pos_ws = receiver_pos_ws;
    return power_light_sampler.eval_area_light_pdf(ctx, LightID(light_id), emitter_pos_ws, emitter_normal_ws);
}

public float
hierarchical_area_light_pdf(uint light_id, float3 receiver_pos_ws, float3 emitter_pos_ws, float3 emitter_normal_ws)
{
    LightSampleContext ctx = {};
    ctx.pos_ws = receiver_pos_ws;
    return hierarchical_light_sampler.eval_area_light_pdf(ctx, LightID(light_id), emitter_pos_ws, emitter_normal_ws);
}

struct AreaLightTraceResult
{
    bool hit;
    float t;
    GeometryInstanceID geometry_instance_id;
};

AreaLightTraceResult trace_area_light_impl(float3 ray_origin_ws, float3 ray_dir_ws, uint instance_inclusion_mask)
{
    const Ray ray = Ray(ray_origin_ws, ray_dir_ws, 0.001, 1000.0);
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(g_scene.tlas, 0, instance_inclusion_mask, ray.to_ray_desc());
    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE)
            continue;
        const GeometryInstanceID geometry_instance_id
            = GeometryInstanceID(query.CandidateInstanceID() + query.CandidateGeometryIndex());
        if (g_scene.get_geometry_instance_type(geometry_instance_id) != GeometryType::area_light)
            continue;
        float t = {};
        if (intersect_area_light(
                geometry_instance_id,
                ray.origin,
                ray.dir,
                query.RayTMin(),
                query.CommittedRayT(),
                t
            )) {
            query.CommitProceduralPrimitiveHit(t);
        }
    }
    AreaLightTraceResult result = {};
    result.hit = query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT;
    if (result.hit) {
        result.t = query.CommittedRayT();
        result.geometry_instance_id = GeometryInstanceID(query.CommittedInstanceID() + query.CommittedGeometryIndex());
    }
    return result;
}

public float4 trace_area_light(float3 ray_origin_ws, float3 ray_dir_ws, uint instance_inclusion_mask)
{
    const AreaLightTraceResult result = trace_area_light_impl(ray_origin_ws, ray_dir_ws, instance_inclusion_mask);
    return result.hit ? float4(
                            1.0,
                            float(uint(GeometryType::area_light)),
                            result.t,
                            uint(get_area_light_id(result.geometry_instance_id))
                        )
                      : float4(0.0);
}

public uint trace_area_light_visible(float3 ray_origin_ws, float3 ray_dir_ws, uint instance_inclusion_mask)
{
    const AreaLightTraceResult result = trace_area_light_impl(ray_origin_ws, ray_dir_ws, instance_inclusion_mask);
    return result.hit && is_area_light_visible(result.geometry_instance_id) ? 1 : 0;
}

public float hierarchical_environment_pdf(float3 direction)
{
    LightSampleContext ctx = {};
    return hierarchical_light_sampler.eval_environment_pdf(ctx, direction);
}

public float hierarchical_emissive_pdf(uint triangle_id)
{
    LightSampleContext ctx = {};
    ctx.pos_ws = float3(0.0, 0.0, 1.0);
    return hierarchical_light_sampler
        .eval_emissive_pdf(ctx, EmissiveTriangleID(triangle_id), float3(0.0), float3(0.0, 0.0, 1.0));
}

public float4 hierarchical_emissive_triangle_tree_sample_pdf(float u)
{
    LightSampleContext ctx = {};
    ctx.pos_ws = float3(0.0, 0.0, 4.0);

    float selection_pdf;
    EmissiveTriangleID triangle_id;
    const bool valid = hierarchical_light_sampler.emissive_triangle_tree.sample(ctx, u, selection_pdf, triangle_id);
    const float evaluated_pdf
        = valid ? hierarchical_light_sampler.emissive_triangle_tree.eval_pdf(ctx, triangle_id) : 0.0;
    return float4(valid ? 1.0 : 0.0, valid ? float(uint(triangle_id)) : -1.0, selection_pdf, evaluated_pdf);
}

public float hierarchical_emissive_triangle_tree_pdf(uint triangle_id)
{
    LightSampleContext ctx = {};
    ctx.pos_ws = float3(0.0, 0.0, 4.0);
    return hierarchical_light_sampler.emissive_triangle_tree.eval_pdf(ctx, EmissiveTriangleID(triangle_id));
}

public float4 hierarchical_emissive_triangle_tree_node_bounds_min_flux(uint node_index)
{
    const EmissiveTriangleTreeNode node = detail::unpack_emissive_triangle_tree_node(
        hierarchical_light_sampler.emissive_triangle_tree.nodes[node_index]
    );
    return float4(node.get_bounds_min(), node.flux);
}

public float4 hierarchical_emissive_triangle_tree_node_bounds_max_cone(uint node_index)
{
    const EmissiveTriangleTreeNode node = detail::unpack_emissive_triangle_tree_node(
        hierarchical_light_sampler.emissive_triangle_tree.nodes[node_index]
    );
    return float4(node.get_bounds_max(), node.cos_cone_angle);
}

public float3 hierarchical_emissive_triangle_tree_node_cone_direction(uint node_index)
{
    return detail::unpack_emissive_triangle_tree_node(
               hierarchical_light_sampler.emissive_triangle_tree.nodes[node_index]
    )
        .cone_direction;
}

public uint2 hierarchical_emissive_triangle_tree_node_leaf_data(uint node_index)
{
    const EmissiveTriangleTreeNode node = detail::unpack_emissive_triangle_tree_node(
        hierarchical_light_sampler.emissive_triangle_tree.nodes[node_index]
    );
    return uint2(node.triangle_offset, node.triangle_count);
}

public uint hierarchical_emissive_triangle_tree_triangle_id(uint triangle_index)
{
    return uint(hierarchical_light_sampler.emissive_triangle_tree.triangle_ids[triangle_index]);
}

public float2 hierarchical_emissive_triangle_tree_hemisphere_pdf(uint triangle_id)
{
    LightSampleContext front_ctx = {};
    front_ctx.pos_ws = float3(0.0, 0.0, 4.0);
    front_ctx.normal_ws = float3(0.0, 0.0, -1.0);

    LightSampleContext back_ctx = {};
    back_ctx.pos_ws = float3(0.0, 0.0, -4.0);
    back_ctx.normal_ws = float3(0.0, 0.0, 1.0);

    return float2(
        hierarchical_light_sampler.emissive_triangle_tree.eval_pdf(front_ctx, EmissiveTriangleID(triangle_id)),
        hierarchical_light_sampler.emissive_triangle_tree.eval_pdf(back_ctx, EmissiveTriangleID(triangle_id))
    );
}

public float hierarchical_emissive_triangle_tree_near_vertex_pdf(uint triangle_id)
{
    LightSampleContext ctx = {};
    ctx.pos_ws = float3(-1.0, -1.0, 1e-30);
    ctx.normal_ws = normalize(float3(1.0, 1.0, 0.0));
    return hierarchical_light_sampler.emissive_triangle_tree.eval_pdf(ctx, EmissiveTriangleID(triangle_id));
}
