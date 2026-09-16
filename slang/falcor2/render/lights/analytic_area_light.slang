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
    const float distance_squared = max(dot(to_light, to_light), 1.0e-9);
    sample.distance = sqrt(distance_squared);
    sample.dir_ws = to_light / sample.distance;

    const float cos_on_light = dot(sampled_normal_ws, -sample.dir_ws);
    if (cos_on_light <= 0.0)
        return false;

    sample.pdf = distance_squared / (cos_on_light * surface_area);
    sample.nee_only = true;
    sample.emitter_type = EmitterType::analytic;
    const float shaping_factor = shaping.eval(light_dir_ws, -sample.dir_ws);
    sample.weight = radiance * shaping_factor / sample.pdf;
    return true;
}
