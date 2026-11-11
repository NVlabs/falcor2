// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct DistantLight : ILight
{
    public LightHeader header;
    public float3x3 world_from_object;
    public float3 radiance;
    public float cos_cutoff_angle;

    public float power_estimate()
    {
        const float solid_angle = float.TWO_PI * (1.0 - cos_cutoff_angle);
        return solid_angle * luminance(radiance);
    }

    private bool is_direction_in_cone(const float3 dir_os)
    {
        // Near either pole, normalized z cannot resolve small angular differences.
        // Compare transverse and axial lengths, preserving the hemisphere before squaring.
        const float sin_cutoff_squared = (1.0 - cos_cutoff_angle) * (1.0 + cos_cutoff_angle);
        const float transverse = dot(dir_os.xy, dir_os.xy) * (cos_cutoff_angle * cos_cutoff_angle);
        const float axial = dir_os.z * dir_os.z * sin_cutoff_squared;
        const bool forward = dir_os.z >= 0.0;
        return select(
            cos_cutoff_angle >= 0.0,
            select(forward, transverse <= axial, false),
            select(forward, true, transverse >= axial)
        );
    }

    public override float3 eval_environment_radiance(const LightSampleContext ctx, const float3 dir_ws)
    {
        const float3 dir_os = mul(transpose(world_from_object), dir_ws);
        if (!is_direction_in_cone(dir_os))
            return float3(0.0);
        return radiance;
    }

    public bool
    sample(const LightSampleContext ctx, const float2 u, out LightSample sample, out LightSampleRef sample_ref)
    {
        sample = {};
        sample_ref = {};

        float3 dir_os = square_to_uniform_cone_concentric(u, cos_cutoff_angle);
        sample.dir_ws = normalize(mul(world_from_object, dir_os));
        sample.distance = float.MAX;
        sample.pdf = square_to_uniform_cone_concentric_pdf(dir_os, cos_cutoff_angle);
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
        const float3 dir_os = mul(transpose(world_from_object), dir_ws);
        if (!is_direction_in_cone(dir_os))
            return 0.0;
        return square_to_uniform_cone_concentric_pdf(dir_os, cos_cutoff_angle);
    }
};
