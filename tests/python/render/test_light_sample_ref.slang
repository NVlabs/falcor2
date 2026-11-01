// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

public ParameterBlock<UniformLightSampler> uniform_sampler;
public ParameterBlock<PowerLightSampler> power_sampler;
public ParameterBlock<HierarchicalLightSampler> hierarchical_sampler;

float relative_error(float3 a, float3 b)
{
    const float3 error = abs(a - b) / max(float3(1.0), max(abs(a), abs(b)));
    return max(error.x, max(error.y, error.z));
}

float sample_error(LightSample a, LightSample b)
{
    float error = max(relative_error(a.radiance, b.radiance), relative_error(a.weight, b.weight));
    error = max(error, relative_error(a.dir_ws, b.dir_ws));
    error = max(error, relative_error(a.pos_ws, b.pos_ws));
    error = max(error, relative_error(a.normal_ws, b.normal_ws));
    error = max(error, relative_error(float3(a.pdf, a.distance, 0.0), float3(b.pdf, b.distance, 0.0)));
    return a.flags == b.flags && a.emitter_type == b.emitter_type && a.emitter_id == b.emitter_id ? error : 1.0;
}

bool evaluate(uint kind, uint light_id, LightSampleContext ctx, LightSampleRef sample_ref, out LightSample sample)
{
    if (kind == 0)
        return g_scene.light_system.get_light(LightID(light_id)).eval_sample(ctx, sample_ref, sample);
    if (kind == 1)
        return uniform_sampler.eval_sample(ctx, sample_ref, sample);
    if (kind == 2)
        return power_sampler.eval_sample(ctx, sample_ref, sample);
    return hierarchical_sampler.eval_sample(ctx, sample_ref, sample);
}

float4 check_samples(
    uint kind,
    uint light_id,
    LightSampleContext ctx,
    float3 other_receiver,
    bool valid,
    bool legacy_valid,
    LightSample sample,
    LightSample legacy_sample,
    LightSampleRef sample_ref
)
{
    float failures = valid == legacy_valid ? 0.0 : 1.0;
    if (!valid)
        return float4(0.0, 0.0, 0.0, failures);
    float error = sample_error(sample, legacy_sample);
    if (!sample_ref.is_valid())
        return float4(-1.0, error, 0.0, failures);

    LightSample repeated = {};
    if (!evaluate(kind, light_id, ctx, sample_ref, repeated))
        failures += 2.0;
    error = max(error, sample_error(sample, repeated));
    ctx.pos_ws = other_receiver;
    LightSample moved = {};
    if (!evaluate(kind, light_id, ctx, sample_ref, moved))
        failures += 4.0;
    float moved_error = 0.0;
    if (sample.emitter_type == EmitterType::environment) {
        moved_error = relative_error(sample.dir_ws, moved.dir_ws);
        if (kind != 0) {
            moved_error = max(
                moved_error,
                relative_error(moved.radiance, g_scene.light_system.eval_environment_radiance(ctx, moved.dir_ws))
            );
        }
    } else {
        moved_error = relative_error(sample.pos_ws, moved.pos_ws);
        moved_error = max(moved_error, relative_error(sample.normal_ws, moved.normal_ws));
        const float3 to_light = sample.pos_ws - other_receiver;
        moved_error = max(moved_error, relative_error(moved.dir_ws, normalize(to_light)));
        moved_error = max(moved_error, relative_error(float3(moved.distance), float3(length(to_light))));
        if (sample.has_flag(LightSampleFlags::has_surface_point)) {
            float pdf;
            if (sample.emitter_type == EmitterType::analytic) {
                const LightID id = LightID(kind == 0 ? light_id : sample_ref.emitter_id);
                let light = g_scene.light_system.get_light(id);
                moved_error = max(
                    moved_error,
                    relative_error(
                        moved.radiance,
                        light.eval_surface_radiance(ctx.pos_ws, moved.pos_ws, moved.normal_ws)
                    )
                );
                if (kind == 0)
                    pdf = light.eval_surface_sampling_pdf(ctx, moved.pos_ws, moved.normal_ws);
                else if (kind == 1)
                    pdf = uniform_sampler.eval_area_light_pdf(ctx, id, moved.pos_ws, moved.normal_ws);
                else if (kind == 2)
                    pdf = power_sampler.eval_area_light_pdf(ctx, id, moved.pos_ws, moved.normal_ws);
                else
                    pdf = hierarchical_sampler.eval_area_light_pdf(ctx, id, moved.pos_ws, moved.normal_ws);
            } else {
                const EmissiveTriangleID id = EmissiveTriangleID(sample_ref.emitter_id);
                if (kind == 1)
                    pdf = uniform_sampler.eval_emissive_pdf(ctx, id, moved.pos_ws, moved.normal_ws);
                else if (kind == 2)
                    pdf = power_sampler.eval_emissive_pdf(ctx, id, moved.pos_ws, moved.normal_ws);
                else
                    pdf = hierarchical_sampler.eval_emissive_pdf(ctx, id, moved.pos_ws, moved.normal_ws);
            }
            moved_error = max(moved_error, relative_error(float3(moved.pdf), float3(pdf)));
        }
    }
    const float3 expected_weight = moved.pdf > 0.0 ? moved.radiance / moved.pdf : float3(0.0);
    moved_error = max(moved_error, relative_error(moved.weight, expected_weight));
    if (!all(isfinite(float4(error, moved_error, moved.pdf, moved.distance))))
        failures += 8.0;
    return float4(1.0, error, moved_error, failures);
}

float4 probe_sampler<
    Sampler : ILightSampler>(Sampler sampler, uint kind, uint seed, LightSampleContext ctx, float3 other_receiver)
{
    UniformSampleGenerator sg = UniformSampleGenerator(seed);
    const LightSampleInput input = make_light_sample_input(sg);
    LightSample sample = {};
    LightSample legacy_sample = {};
    LightSampleRef sample_ref = {};
    const bool valid = sampler.sample(ctx, input, sample, sample_ref);
    const bool legacy_valid = sampler.sample(ctx, input, legacy_sample);
    return check_samples(kind, 0, ctx, other_receiver, valid, legacy_valid, sample, legacy_sample, sample_ref);
}

public float4 probe(uint seed, uint kind, uint light_id, float3 receiver, float3 other_receiver)
{
    LightSampleContext ctx = {};
    ctx.pos_ws = receiver;
    if (kind == 1)
        return probe_sampler<UniformLightSampler>(uniform_sampler, kind, seed, ctx, other_receiver);
    if (kind == 2)
        return probe_sampler<PowerLightSampler>(power_sampler, kind, seed, ctx, other_receiver);
    if (kind == 3)
        return probe_sampler<HierarchicalLightSampler>(hierarchical_sampler, kind, seed, ctx, other_receiver);
    UniformSampleGenerator sg = UniformSampleGenerator(seed);
    const float2 u = next_float2(sg);
    LightSample sample = {};
    LightSample legacy_sample = {};
    LightSampleRef sample_ref = {};
    let light = g_scene.light_system.get_light(LightID(light_id));
    const bool valid = light.sample(ctx, u, sample, sample_ref);
    const bool legacy_valid = light.sample(ctx, u, legacy_sample);
    return check_samples(kind, light_id, ctx, other_receiver, valid, legacy_valid, sample, legacy_sample, sample_ref);
}

public float4 evaluate_reference(
    uint field,
    uint kind,
    uint emitter_type,
    uint emitter_id,
    float2 parameters,
    float3 receiver,
    float3 normal
)
{
    if (field == 4)
        return float4(sizeof(LightSampleRef), 0.0, 0.0, 0.0);
    LightSampleRef sample_ref = {};
    sample_ref.emitter_type = EmitterType(emitter_type);
    sample_ref.emitter_id = emitter_id;
    sample_ref.parameters = parameters;
    LightSampleContext ctx = {};
    ctx.pos_ws = receiver;
    ctx.normal_ws = normal;
    LightSample sample = {};
    const bool valid = evaluate(kind, emitter_id, ctx, sample_ref, sample);
    if (field == 0)
        return float4(sample.radiance, valid ? 1.0 : 0.0);
    if (field == 1)
        return float4(sample.weight, sample.pdf);
    if (field == 2)
        return float4(sample.pos_ws, sample.distance);
    return float4(sample.dir_ws, uint(sample.flags));
}
