// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct SphereLight : ILight
{
    public LightHeader header;
    public float3 center_ws;
    public float radius_ws;
    public float3 dir_ws;
    public float3 radiance;
    public float surface_area;
    public bool enable_virtual_sphere_shrinking;
    public LightShaping shaping;

    public float power_estimate()
    {
        return luminance(radiance) * float.PI * surface_area * shaping.full_sphere_power_factor();
    }

    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    )
    {
        sample = {};
        if (radius_ws <= 0.0)
            return false;

        float effective_radius_ws = radius_ws;
        float effective_surface_area = surface_area;
        float3 effective_radiance = radiance;
        if (enable_virtual_sphere_shrinking) {
            const float receiver_distance_squared = dot(ctx.pos_ws - center_ws, ctx.pos_ws - center_ws);
            const float authored_radius_squared = radius_ws * radius_ws;
            const float effective_radius_squared = min(authored_radius_squared, 0.99 * receiver_distance_squared);
            if (effective_radius_squared <= 1.0e-12)
                return false;

            const float area_scale = effective_radius_squared / authored_radius_squared;
            effective_radius_ws = sqrt(effective_radius_squared);
            effective_surface_area *= area_scale;
            effective_radiance /= area_scale;
        }

        const float2 u = next_float2(sg);
        const float3 pos_on_unit_sphere = square_to_uniform_sphere(u.yx);
        const float3 sampled_pos_ws = center_ws + effective_radius_ws * pos_on_unit_sphere;

        return finalize_analytic_area_light_sample(
            ctx.pos_ws,
            sampled_pos_ws,
            pos_on_unit_sphere,
            effective_surface_area,
            effective_radiance,
            shaping,
            dir_ws,
            sample
        );
    }
};
