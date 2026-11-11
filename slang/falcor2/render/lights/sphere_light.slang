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

    /// Returns true when the receiver uses the NEE-only virtual sampling proxy instead of the authored surface.
    private bool uses_virtual_sampling_proxy(const float3 receiver_pos_ws)
    {
        if (!enable_virtual_sphere_shrinking || radius_ws <= 0.0)
            return false;
        const float3 center_to_receiver = receiver_pos_ws - center_ws;
        return radius_ws * radius_ws > 0.99 * dot(center_to_receiver, center_to_receiver);
    }

    /// Computes the solid-angle cone subtended by a sphere at a receiver outside it.
    private bool get_sampling_cone(
        const float3 receiver_pos_ws,
        const float sphere_radius_ws,
        out float3 cone_axis_ws,
        out float cos_cone_angle
    )
    {
        cone_axis_ws = {};
        cos_cone_angle = {};
        const float3 receiver_to_center = center_ws - receiver_pos_ws;
        const float receiver_distance_squared = dot(receiver_to_center, receiver_to_center);
        const float sphere_radius_squared = sphere_radius_ws * sphere_radius_ws;
        if (sphere_radius_squared <= 0.0 || receiver_distance_squared <= sphere_radius_squared)
            return false;
        cone_axis_ws = receiver_to_center * rsqrt(receiver_distance_squared);
        cos_cone_angle = safe_sqrt(1.0 - sphere_radius_squared / receiver_distance_squared);
        return true;
    }

    public bool
    sample(const LightSampleContext ctx, const float2 u, out LightSample sample, out LightSampleRef sample_ref)
    {
        sample = {};
        sample_ref = {};

        if (radius_ws <= 0.0)
            return false;

        float effective_radius_ws = radius_ws;
        float3 effective_radiance = radiance;
        const bool uses_virtual_proxy = uses_virtual_sampling_proxy(ctx.pos_ws);
        if (uses_virtual_proxy) {
            const float receiver_distance_squared = dot(ctx.pos_ws - center_ws, ctx.pos_ws - center_ws);
            const float authored_radius_squared = radius_ws * radius_ws;
            const float effective_radius_squared = 0.99 * receiver_distance_squared;
            if (effective_radius_squared <= 1.0e-12)
                return false;

            const float area_scale = effective_radius_squared / authored_radius_squared;
            effective_radius_ws = sqrt(effective_radius_squared);
            effective_radiance /= area_scale;
        }

        float3 cone_axis_ws;
        float cos_cone_angle;
        if (!get_sampling_cone(ctx.pos_ws, effective_radius_ws, cone_axis_ws, cos_cone_angle))
            return false;

        const float3 sample_dir_local = square_to_uniform_cone_concentric(u, cos_cone_angle);
        sample.dir_ws = Frame(cone_axis_ws).to_global(sample_dir_local);
        if (!intersect_ray_sphere(ctx.pos_ws, sample.dir_ws, center_ws, effective_radius_ws, sample.distance))
            return false;

        sample.pos_ws = ctx.pos_ws + sample.distance * sample.dir_ws;
        sample.normal_ws = (sample.pos_ws - center_ws) / effective_radius_ws;
        const float shaping_factor = shaping.eval(dir_ws, -sample.dir_ws);
        if (shaping_factor <= 0.0)
            return false;

        sample.pdf = square_to_uniform_cone_concentric_pdf(sample_dir_local, cos_cone_angle);
        sample.radiance = effective_radiance * shaping_factor;
        sample.weight = sample.radiance / sample.pdf;
        // Only the shrunken proxy lacks a matching surface-hit strategy. Samples
        // on the authored sphere participate in MIS with eval_surface_sampling_pdf().
        sample.flags = LightSampleFlags::has_surface_point;
        if (uses_virtual_proxy)
            sample.flags |= LightSampleFlags::nee_only;
        sample.emitter_type = EmitterType::analytic;

        if (!uses_virtual_proxy) {
            sample_ref.emitter_type = EmitterType::analytic;
            sample_ref.parameters = ndir_to_oct_unorm(sample.normal_ws);
        }

        return true;
    }

    public bool eval_sample(const LightSampleContext ctx, const LightSampleRef sample_ref, out LightSample sample)
    {
        sample = {};
        if (uses_virtual_sampling_proxy(ctx.pos_ws))
            return false;
        const float3 normal = oct_to_ndir_unorm(sample_ref.parameters);
        return eval_analytic_light_sample(this, ctx, sample_ref, center_ws + radius_ws * normal, normal, sample);
    }

    public override bool intersect_surface(const Ray ray, out float t)
    {
        t = {};
        if (radius_ws <= 0.0 || uses_virtual_sampling_proxy(ray.origin))
            return false;
        return intersect_ray_sphere(ray.origin, ray.dir, center_ws, radius_ws, t, ray.t_min, ray.t_max);
    }

    public override float3 eval_surface_normal(const float3 pos_ws)
    {
        const float3 offset = pos_ws - center_ws;
        const float length_squared = dot(offset, offset);
        return length_squared > 0.0 ? offset * rsqrt(length_squared) : float3(0.0);
    }

    public override float3
    eval_surface_radiance(const float3 receiver_pos_ws, const float3 emitter_pos_ws, const float3 emitter_normal_ws)
    {
        if (uses_virtual_sampling_proxy(receiver_pos_ws))
            return {};
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
        if (radius_ws <= 0.0 || uses_virtual_sampling_proxy(ctx.pos_ws))
            return 0.0;

        float3 cone_axis_ws;
        float cos_cone_angle;
        if (!get_sampling_cone(ctx.pos_ws, radius_ws, cone_axis_ws, cos_cone_angle))
            return 0.0;

        const float3 to_emitter = emitter_pos_ws - ctx.pos_ws;
        const float distance_squared = dot(to_emitter, to_emitter);
        if (distance_squared <= 0.0 || dot(emitter_normal_ws, -to_emitter) <= 0.0)
            return 0.0;
        const float3 sample_dir_local = Frame(cone_axis_ws).to_local(to_emitter * rsqrt(distance_squared));
        return square_to_uniform_cone_concentric_pdf(sample_dir_local, cos_cone_angle, true);
    }
};
