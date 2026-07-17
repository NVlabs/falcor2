// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct SpotLight : ILight
{
    public LightHeader header;
    public float3 position_ws;
    public float3 dir_ws;
    public float3 intensity;
    public float cos_cutoff_angle;
    public float cos_falloff_angle;

    public float emitted_power()
    {
        const float solid_angle_factor = (1.0 - cos_falloff_angle) + (cos_falloff_angle - cos_cutoff_angle) / 5.0;
        return float.TWO_PI * solid_angle_factor * luminance(intensity);
    }

    public float3 eval_direction<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        return float3(0.0);
    }

    public bool sample_direction<Interaction : IInteraction, SampleGenerator : ISampleGenerator>(
        const Interaction ref,
        inout SampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    )
    {
        sample = {};
        float3 to_light = position_ws - ref.position_ws;
        float distance = length(to_light);
        sample.dir_ws = to_light / distance;
        sample.distance = distance;
        sample.pdf = 1.0;
        sample.delta = true;
        sample.emitter_type = EmitterType::analytic;
        weight = intensity * falloff(-sample.dir_ws) / sqr(distance);
        return true;
    }

    public float eval_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        return 0.0;
    }

    internal float falloff(float3 dir)
    {
        float cos_theta = dot(dir_ws, dir);
        if (cos_theta < cos_cutoff_angle)
            return 0.0;
        if (cos_theta > cos_falloff_angle)
            return 1.0;
        float delta = (cos_theta - cos_cutoff_angle) / (cos_falloff_angle - cos_cutoff_angle);
        return sqr(sqr(delta));
    }
};
