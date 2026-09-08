// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct ConstantLight : ILight
{
    public LightHeader header;
    public float3 radiance;

    public float power_estimate() { return float.FOUR_PI * luminance(radiance); }

    public override float3 eval_environment_radiance(const LightSampleContext ctx, const float3 dir_ws)
    {
        return radiance;
    }

    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    )
    {
        sample = {};
        sample.dir_ws = square_to_uniform_sphere(next_float2(sg));
        sample.distance = float.MAX;
        sample.pdf = square_to_uniform_sphere_pdf(sample.dir_ws);
        sample.weight = radiance / sample.pdf;
        sample.emitter_type = EmitterType::environment;
        return true;
    }

    public override float eval_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        return square_to_uniform_sphere_pdf(dir_ws);
    }
};
