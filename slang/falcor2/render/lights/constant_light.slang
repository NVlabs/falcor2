// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct ConstantLight : ILight
{
    public LightHeader header;
    public float3 radiance;

    public float3 eval_direction<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
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
        sample.dir_ws = square_to_uniform_sphere(next_float2(sg));
        sample.distance = float.MAX;
        sample.pdf = square_to_uniform_sphere_pdf(sample.dir_ws);
        sample.emitter_type = EmitterType::environment;
        weight = radiance / sample.pdf;
        return true;
    }

    public float eval_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        return square_to_uniform_sphere_pdf(sample.dir_ws);
    }
};
