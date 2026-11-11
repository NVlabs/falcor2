// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct DiskLight : ILight
{
    public LightHeader header;
    public float3 center_ws;
    public float3 radius_u_ws;
    public float3 radius_v_ws;
    public float3 normal_ws;
    public float3 radiance;
    public float surface_area;
    public LightShaping shaping;

    public float power_estimate()
    {
        return luminance(radiance) * float.PI * surface_area * shaping.cosine_hemisphere_power_factor();
    }

    public bool
    sample(const LightSampleContext ctx, const float2 u, out LightSample sample, out LightSampleRef sample_ref)
    {
        sample = {};
        sample_ref = {};

        if (surface_area <= 0.0 || dot(normal_ws, ctx.pos_ws - center_ws) <= 0.0)
            return false;

        const float2 pos_on_unit_disk = square_to_uniform_disk_polar(u.yx);
        const float3 sampled_pos_ws = center_ws + pos_on_unit_disk.x * radius_u_ws + pos_on_unit_disk.y * radius_v_ws;

        sample_ref.emitter_type = EmitterType::analytic;
        sample_ref.parameters = pos_on_unit_disk;
        return finalize_analytic_area_light_sample(
            ctx.pos_ws,
            sampled_pos_ws,
            normal_ws,
            surface_area,
            radiance,
            shaping,
            normal_ws,
            sample
        );
    }

    public bool eval_sample(const LightSampleContext ctx, const LightSampleRef sample_ref, out LightSample sample)
    {
        sample = {};
        const float2 p = sample_ref.parameters;
        return eval_analytic_light_sample(
            this,
            ctx,
            sample_ref,
            center_ws + p.x * radius_u_ws + p.y * radius_v_ws,
            normal_ws,
            sample
        );
    }

    public override bool intersect_surface(const Ray ray, out float t)
    {
        return intersect_analytic_planar_light(ray, center_ws, radius_u_ws, radius_v_ws, normal_ws, true, t);
    }

    public override float3 eval_surface_normal(const float3 pos_ws) { return normal_ws; }

    public override float3
    eval_surface_radiance(const float3 receiver_pos_ws, const float3 emitter_pos_ws, const float3 emitter_normal_ws)
    {
        return eval_analytic_area_light_surface_radiance(
            receiver_pos_ws,
            emitter_pos_ws,
            emitter_normal_ws,
            radiance,
            shaping,
            normal_ws
        );
    }

    public override float
    eval_surface_sampling_pdf(const LightSampleContext ctx, const float3 emitter_pos_ws, const float3 emitter_normal_ws)
    {
        return eval_area_sample_solid_angle_pdf(ctx.pos_ws, emitter_pos_ws, emitter_normal_ws, surface_area);
    }
};
