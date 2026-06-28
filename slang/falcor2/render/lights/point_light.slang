// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct PointLight : ILight
{
    public LightHeader header;
    public float3 position_ws;
    public float3 intensity;

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
        weight = intensity / sqr(distance);
        return true;
    }

    public float eval_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        return 0.0;
    }
};
