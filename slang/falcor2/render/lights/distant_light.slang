// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct DistantLight : ILight
{
    public LightHeader header;
    public float3x3 world_from_object;
    public float3 radiance;
    public float cos_cutoff_angle;

    public float emitted_power()
    {
        const float solid_angle = float.TWO_PI * (1.0 - cos_cutoff_angle);
        return solid_angle * luminance(radiance);
    }

    public float3 eval_direction<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        float3 dir_os = normalize(mul(transpose(world_from_object), sample.dir_ws));
        if (dir_os.z < cos_cutoff_angle)
            return float3(0.0);
        return radiance;
    }

    public bool sample_direction<Interaction : IInteraction, SampleGenerator : ISampleGenerator>(
        const Interaction ref,
        inout SampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    )
    {
        sample = {};
        float3 dir_os = square_to_uniform_cone_concentric(next_float2(sg), cos_cutoff_angle);
        sample.dir_ws = normalize(mul(world_from_object, dir_os));
        sample.distance = float.MAX;
        sample.pdf = square_to_uniform_cone_concentric_pdf(dir_os, cos_cutoff_angle);
        sample.emitter_type = EmitterType::environment;
        weight = radiance / sample.pdf;
        return true;
    }

    public float eval_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        float3 dir_os = normalize(mul(transpose(world_from_object), sample.dir_ws));
        if (dir_os.z < cos_cutoff_angle)
            return 0.0;
        return square_to_uniform_cone_concentric_pdf(dir_os, cos_cutoff_angle);
    }
};
