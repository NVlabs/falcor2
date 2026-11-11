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

    public bool
    sample(const LightSampleContext ctx, const float2 u, out LightSample sample, out LightSampleRef sample_ref)
    {
        sample = {};
        sample_ref = {};

        sample.dir_ws = square_to_uniform_sphere(u);
        sample.distance = float.MAX;
        sample.pdf = square_to_uniform_sphere_pdf(sample.dir_ws);
        sample.radiance = radiance;
        sample.weight = sample.radiance / sample.pdf;
        sample.emitter_type = EmitterType::environment;

        sample_ref.emitter_type = EmitterType::environment;
        sample_ref.parameters = ndir_to_oct_unorm(sample.dir_ws);

        return true;
    }

    public bool eval_sample(const LightSampleContext ctx, const LightSampleRef sample_ref, out LightSample sample)
    {
        return eval_environment_light_sample(this, ctx, sample_ref, sample);
    }

    public override float eval_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        return square_to_uniform_sphere_pdf(dir_ws);
    }
};
