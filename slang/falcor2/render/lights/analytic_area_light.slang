// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Finalizes a uniformly sampled point on a one-sided analytic area light.
/// The returned PDF is with respect to solid angle at the shading point.
public bool finalize_analytic_area_light_sample(
    const float3 shading_pos_ws,
    const float3 sampled_pos_ws,
    const float3 sampled_normal_ws,
    const float surface_area,
    const float3 radiance,
    const LightShaping shaping,
    const float3 light_dir_ws,
    out LightSample sample
)
{
    sample = {};

    if (surface_area <= 0.0)
        return false;

    sample.pos_ws = sampled_pos_ws;
    sample.normal_ws = sampled_normal_ws;

    const float3 to_light = sampled_pos_ws - shading_pos_ws;
    const float cosine_numerator = dot(sampled_normal_ws, -to_light);
    if (cosine_numerator <= 0.0)
        return false;

    const float distance_squared = max(dot(to_light, to_light), 1.0e-9);
    sample.distance = sqrt(distance_squared);
    sample.dir_ws = to_light / sample.distance;

    const float shaping_factor = shaping.eval(light_dir_ws, -sample.dir_ws);
    if (shaping_factor <= 0.0)
        return false;

    const float cos_on_light = cosine_numerator / sample.distance;
    const float geometry_term = cos_on_light * surface_area / distance_squared;
    sample.pdf = 1.0 / geometry_term;
    sample.weight = radiance * shaping_factor * geometry_term;
    // Disk and rectangle samples lie on the authored surface and have an exact
    // complementary surface-hit PDF, so they participate in MIS.
    sample.flags = LightSampleFlags::has_surface_point;
    sample.emitter_type = EmitterType::analytic;
    return true;
}

/// Evaluates radiance emitted by a shaped, one-sided analytic area light toward a receiver.
public float3 eval_analytic_area_light_surface_radiance(
    const float3 receiver_pos_ws,
    const float3 emitter_pos_ws,
    const float3 emitter_normal_ws,
    const float3 radiance,
    const LightShaping shaping,
    const float3 light_dir_ws
)
{
    const float3 to_receiver = receiver_pos_ws - emitter_pos_ws;
    const float distance_squared = dot(to_receiver, to_receiver);
    if (distance_squared <= 0.0 || dot(emitter_normal_ws, to_receiver) <= 0.0)
        return {};
    return radiance * shaping.eval(light_dir_ws, to_receiver * rsqrt(distance_squared));
}

/// Converts a uniform area-sampling density to solid-angle measure at the receiver.
public float eval_area_sample_solid_angle_pdf(
    const float3 receiver_pos_ws,
    const float3 emitter_pos_ws,
    const float3 emitter_normal_ws,
    const float surface_area
)
{
    if (surface_area <= 0.0)
        return 0.0;
    const float3 to_light = emitter_pos_ws - receiver_pos_ws;
    const float distance_squared = dot(to_light, to_light);
    if (distance_squared <= 0.0)
        return 0.0;
    const float cos_on_light = dot(emitter_normal_ws, -to_light * rsqrt(distance_squared));
    return cos_on_light > 0.0 ? distance_squared / (cos_on_light * surface_area) : 0.0;
}

/// Intersects a planar light parameterized by two world-space radius or half-extent vectors.
/// Set elliptical to true for a disk and false for a rectangle.
public bool intersect_analytic_planar_light(
    const Ray ray,
    const float3 center_ws,
    const float3 axis_u_ws,
    const float3 axis_v_ws,
    const float3 normal_ws,
    const bool elliptical,
    out float t
)
{
    t = {};
    const float denominator = dot(ray.dir, normal_ws);
    if (abs(denominator) <= float.MIN)
        return false;

    t = dot(center_ws - ray.origin, normal_ws) / denominator;
    if (t < ray.t_min || t > ray.t_max)
        return false;

    const float3 offset = ray.origin + t * ray.dir - center_ws;
    const float uu = dot(axis_u_ws, axis_u_ws);
    const float uv = dot(axis_u_ws, axis_v_ws);
    const float vv = dot(axis_v_ws, axis_v_ws);
    const float determinant = uu * vv - uv * uv;
    if (determinant <= float.MIN)
        return false;
    const float ou = dot(offset, axis_u_ws);
    const float ov = dot(offset, axis_v_ws);
    const float u = (vv * ou - uv * ov) / determinant;
    const float v = (uu * ov - uv * ou) / determinant;
    return elliptical ? u * u + v * v <= 1.0 : max(abs(u), abs(v)) <= 1.0;
}
