// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct PointLight : ILight
{
    public LightHeader header;
    public float3 pos_ws;
    public float3 dir_ws;
    public float3 intensity;
    public LightShaping shaping;

    public float power_estimate() { return float.FOUR_PI * luminance(intensity) * shaping.full_sphere_power_factor(); }

    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    )
    {
        sample = {};
        float3 to_light = pos_ws - ctx.pos_ws;
        float distance = length(to_light);
        if (distance <= 0.0)
            return false;
        sample.dir_ws = to_light / distance;
        sample.distance = distance;
        sample.pdf = 1.0;
        sample.weight = intensity * shaping.eval(dir_ws, -sample.dir_ws) / sqr(distance);
        sample.nee_only = true;
        sample.emitter_type = EmitterType::analytic;
        sample.pos_ws = pos_ws;
        return true;
    }
};
