// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct PointLight : ILight
{
    public LightHeader header;
    public float3 pos_ws;
    public float3 dir_ws;
    public float3 radiant_intensity;
    public LightShaping shaping;

    public float power_estimate()
    {
        return float.FOUR_PI * luminance(radiant_intensity) * shaping.full_sphere_power_factor();
    }

    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    )
    {
        sample = {};
        const float3 to_light = pos_ws - ctx.pos_ws;
        const float distance_squared = dot(to_light, to_light);
        if (distance_squared <= 0.0)
            return false;

        const float distance = sqrt(distance_squared);
        const float3 sample_dir_ws = to_light / distance;
        const float shaping_factor = shaping.eval(dir_ws, -sample_dir_ws);
        if (shaping_factor <= 0.0)
            return false;

        sample.dir_ws = sample_dir_ws;
        sample.distance = distance;
        sample.pdf = 1.0;
        sample.weight = radiant_intensity * shaping_factor / distance_squared;
        sample.flags = LightSampleFlags::nee_only;
        sample.emitter_type = EmitterType::analytic;
        sample.pos_ws = pos_ws;
        return true;
    }
};
