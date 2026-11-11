// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Finite circular cylinder that emits from its open lateral surface.
public struct CylinderLight : ILight
{
    public LightHeader header;
    public float3 center_ws;
    public float3 axis_ws;
    public float3 radius_u_ws;
    public float3 radius_v_ws;
    public float half_length_ws;
    public float radius_ws;
    public float3 dir_ws;
    public float3 radiance;
    public float surface_area;
    public LightShaping shaping;

    public float power_estimate()
    {
        // Shaping is fixed around the light's local -Z axis while cylinder
        // normals vary azimuthally. Its sphere-average is a stable selection
        // estimate; exact directional evaluation below remains unbiased.
        return luminance(radiance) * float.PI * surface_area * shaping.full_sphere_power_factor();
    }

    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    )
    {
        sample = {};
        if (radius_ws <= 0.0 || half_length_ws <= 0.0 || surface_area <= 0.0)
            return false;

        const float2 u = next_float2(sg);
        float sin_phi;
        float cos_phi;
        sincos(float.TWO_PI* u.x, sin_phi, cos_phi);
        const float3 sampled_normal_ws = (cos_phi * radius_u_ws + sin_phi * radius_v_ws) / radius_ws;
        const float3 sampled_pos_ws
            = center_ws + (2.0 * u.y - 1.0) * half_length_ws * axis_ws + radius_ws * sampled_normal_ws;
        return finalize_analytic_area_light_sample(
            ctx.pos_ws,
            sampled_pos_ws,
            sampled_normal_ws,
            surface_area,
            radiance,
            shaping,
            dir_ws,
            sample
        );
    }

    public override bool intersect_surface(const Ray ray, out float t)
    {
        t = {};
        if (radius_ws <= 0.0 || half_length_ws <= 0.0)
            return false;
        return intersect_ray_open_cylinder(
            ray.origin,
            ray.dir,
            center_ws,
            axis_ws,
            radius_ws,
            half_length_ws,
            t,
            ray.t_min,
            ray.t_max
        );
    }

    public override float3 eval_surface_normal(const float3 pos_ws)
    {
        const float3 offset = pos_ws - center_ws;
        const float3 radial = offset - dot(offset, axis_ws) * axis_ws;
        const float radial_length_squared = dot(radial, radial);
        return radial_length_squared > 0.0 ? radial * rsqrt(radial_length_squared) : float3(0.0);
    }

    public override float3
    eval_surface_radiance(const float3 receiver_pos_ws, const float3 emitter_pos_ws, const float3 emitter_normal_ws)
    {
        return eval_analytic_area_light_surface_radiance(
            receiver_pos_ws,
            emitter_pos_ws,
            emitter_normal_ws,
            radiance,
            shaping,
            dir_ws
        );
    }

    public override float
    eval_surface_sampling_pdf(const LightSampleContext ctx, const float3 emitter_pos_ws, const float3 emitter_normal_ws)
    {
        return eval_area_sample_solid_angle_pdf(ctx.pos_ws, emitter_pos_ws, emitter_normal_ws, surface_area);
    }
};
