// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct RectLight : ILight
{
    public LightHeader header;
    public float3 center_ws;
    public float3 half_extent_u_ws;
    public float3 half_extent_v_ws;
    public float3 normal_ws;
    public float3 radiance;
    public float surface_area;
    public LightShaping shaping;

    public float power_estimate()
    {
        return luminance(radiance) * float.PI * surface_area * shaping.cosine_hemisphere_power_factor();
    }

    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    )
    {
        const float2 offset = 2.0 * next_float2(sg) - 1.0;
        const float3 sampled_pos_ws = center_ws + offset.x * half_extent_u_ws + offset.y * half_extent_v_ws;
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
};
