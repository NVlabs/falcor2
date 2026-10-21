// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
__exported import falcor2.render;
__exported import falcor2.utils;

namespace light_surface_sampling_tests {

public struct AreaLightDirectionTarget : chi2::ITarget
{
    public uint light_id;
    public float3 receiver_pos_ws;
    public float2 domain_min;
    public float2 domain_max;

    typedef chi2::SphericalDomain Domain;

    public Domain get_domain() { return chi2::SphericalDomain(domain_min, domain_max); }

    public bool sample<SG : ISampleGenerator>(inout SG sg, out float3 p)
    {
        LightSampleContext ctx = {};
        ctx.pos_ws = receiver_pos_ws;
        LightSample sample = {};
        const bool valid = g_scene.light_system.get_light(LightID(light_id)).sample(ctx, sg, sample);
        p = sample.dir_ws;
        return valid;
    }

    public float pdf(float3 dir_ws)
    {
        ILight light = g_scene.light_system.get_light(LightID(light_id));
        float t;
        if (!light.intersect_surface(Ray(receiver_pos_ws, dir_ws, 0.0, float.INFINITY), t))
            return 0.0;
        const float3 emitter_pos_ws = receiver_pos_ws + t * dir_ws;
        LightSampleContext ctx = {};
        ctx.pos_ws = receiver_pos_ws;
        const float3 emitter_normal_ws = light.eval_surface_normal(emitter_pos_ws);
        return light.eval_surface_sampling_pdf(ctx, emitter_pos_ws, emitter_normal_ws);
    }
};

public struct CylinderLightSurfaceTarget : chi2::ITarget
{
    public uint light_id;
    public float3 receiver_pos_ws;
    public float3 center_ws;
    public float3 axis_ws;
    public float3 radius_u_ws;
    public float3 radius_v_ws;
    public float half_length_ws;

    typedef chi2::PlanarDomain Domain;

    public Domain get_domain()
    {
        return chi2::PlanarDomain(float2(-float.PI, -half_length_ws), float2(float.PI, half_length_ws));
    }

    public bool sample<SG : ISampleGenerator>(inout SG sg, out float2 p)
    {
        LightSampleContext ctx = {};
        ctx.pos_ws = receiver_pos_ws;
        LightSample sample = {};
        const bool valid = g_scene.light_system.get_light(LightID(light_id)).sample(ctx, sg, sample);
        const float3 offset = sample.pos_ws - center_ws;
        const float3 radial = offset - dot(offset, axis_ws) * axis_ws;
        p = float2(atan2(dot(radial, radius_v_ws), dot(radial, radius_u_ws)), dot(offset, axis_ws));
        return valid;
    }

    public float pdf(float2 p)
    {
        float sin_phi;
        float cos_phi;
        sincos(p.x, sin_phi, cos_phi);
        const float3 normal_ws = normalize(cos_phi * radius_u_ws + sin_phi * radius_v_ws);
        const float radius_ws = length(radius_u_ws);
        const float3 emitter_pos_ws = center_ws + p.y * axis_ws + radius_ws * normal_ws;
        if (dot(normal_ws, receiver_pos_ws - emitter_pos_ws) <= 0.0)
            return 0.0;
        return 1.0 / (float.TWO_PI * (2.0 * half_length_ws));
    }
};

float relative_error(const float a, const float b)
{
    return abs(a - b) / max(max(abs(a), abs(b)), 1e-8);
}

[noinline]
float direction_unit_error(float3 dir_ws)
{
    // R535 CUDA with fast floating point mode faults when the consistency kernel
    // reads sample.dir_ws directly after the dynamic ILight.sample() call.
    return abs(length(dir_ws) - 1.0);
}

public void test_eval_sample_consistency(
    uint tid,
    uint sample_count,
    uint samples_per_thread,
    uint seed,
    uint light_id,
    float3 receiver_pos_ws,
    bool expected_nee_only,
    WTensor<float, 1> pdf_errors,
    WTensor<float, 1> normal_errors,
    WTensor<float, 1> weight_errors,
    WTensor<float, 1> geometry_errors,
    AtomicTensor<uint, 1> counters
)
{
    const uint2 v = block_cipher_tea(seed, tid);
    UniformSampleGenerator sg = UniformSampleGenerator(v.x);
    ILight light = g_scene.light_system.get_light(LightID(light_id));
    LightSampleContext ctx = {};
    ctx.pos_ws = receiver_pos_ws;
    float max_pdf_error = 0.0;
    float max_normal_error = 0.0;
    float max_weight_error = 0.0;
    float max_geometry_error = 0.0;

    for (uint i = 0; i < samples_per_thread; ++i) {
        const uint sample_index = tid * samples_per_thread + i;
        if (sample_index >= sample_count)
            break;

        LightSample sample = {};
        if (!light.sample(ctx, sg, sample)) {
            counters.add(1, 1);
            continue;
        }
        counters.add(0, 1);

        const float3 evaluated_normal_ws = light.eval_surface_normal(sample.pos_ws);
        max_normal_error = max(max_normal_error, length(evaluated_normal_ws - sample.normal_ws));

        const float3 to_light = sample.pos_ws - receiver_pos_ws;
        max_geometry_error = max(max_geometry_error, direction_unit_error(sample.dir_ws));
        max_geometry_error = max(max_geometry_error, length(to_light - sample.distance * sample.dir_ws));

        const float evaluated_pdf = light.eval_surface_sampling_pdf(ctx, sample.pos_ws, evaluated_normal_ws);
        const float cos_on_light = dot(evaluated_normal_ws, -sample.dir_ws);
        // Reconstructing a sphere hit from a rounded direction is ill-conditioned at the silhouette.
        // The direction-space chi-square test covers that region statistically; keep this pointwise check stable.
        if (evaluated_pdf > 0.0) {
            if (cos_on_light > 5e-2) {
                float intersection_t;
                if (!light
                         .intersect_surface(Ray(receiver_pos_ws, sample.dir_ws, 0.0, float.INFINITY), intersection_t)) {
                    counters.add(2, 1);
                } else {
                    max_geometry_error = max(max_geometry_error, relative_error(intersection_t, sample.distance));
                    max_geometry_error = max(
                        max_geometry_error,
                        length(receiver_pos_ws + intersection_t * sample.dir_ws - sample.pos_ws)
                    );
                }
                max_pdf_error = max(max_pdf_error, relative_error(evaluated_pdf, sample.pdf));
                const float3 radiance
                    = light.eval_surface_radiance(receiver_pos_ws, sample.pos_ws, evaluated_normal_ws);
                const float3 expected_weight = radiance / evaluated_pdf;
                for (uint channel = 0; channel < 3; ++channel)
                    max_weight_error
                        = max(max_weight_error, relative_error(expected_weight[channel], sample.weight[channel]));
            }
        }

        if (!sample.has_flag(LightSampleFlags::has_surface_point)
            || sample.has_flag(LightSampleFlags::nee_only) != expected_nee_only
            || sample.emitter_type != EmitterType::analytic || !all(isfinite(sample.pos_ws))
            || !all(isfinite(sample.normal_ws)) || !isfinite(sample.pdf) || !all(isfinite(sample.weight))) {
            counters.add(2, 1);
        }
    }

    pdf_errors.store(tid, max_pdf_error);
    normal_errors.store(tid, max_normal_error);
    weight_errors.store(tid, max_weight_error);
    geometry_errors.store(tid, max_geometry_error);
}

public float4
eval_surface_values(uint light_id, float3 receiver_pos_ws, float3 emitter_pos_ws, float3 emitter_normal_ws)
{
    ILight light = g_scene.light_system.get_light(LightID(light_id));
    LightSampleContext ctx = {};
    ctx.pos_ws = receiver_pos_ws;
    const float pdf = light.eval_surface_sampling_pdf(ctx, emitter_pos_ws, emitter_normal_ws);
    return float4(pdf, light.eval_surface_radiance(receiver_pos_ws, emitter_pos_ws, emitter_normal_ws));
}

public float4 eval_surface_metadata(uint light_id, float3 receiver_pos_ws, float3 emitter_pos_ws)
{
    ILight light = g_scene.light_system.get_light(LightID(light_id));
    LightSampleContext ctx = {};
    ctx.pos_ws = receiver_pos_ws;
    const float3 emitter_normal_ws = light.eval_surface_normal(emitter_pos_ws);
    const float pdf = light.eval_surface_sampling_pdf(ctx, emitter_pos_ws, emitter_normal_ws);
    return float4(emitter_normal_ws, pdf > 0.0 ? 1.0 : 0.0);
}

public float eval_surface_intersection(uint light_id, float3 origin_ws, float3 dir_ws, float t_min, float t_max)
{
    ILight light = g_scene.light_system.get_light(LightID(light_id));
    float t;
    return light.intersect_surface(Ray(origin_ws, dir_ws, t_min, t_max), t) ? t : -1.0;
}

public uint sample_flags(uint light_id, float3 receiver_pos_ws)
{
    LightSampleContext ctx = {};
    ctx.pos_ws = receiver_pos_ws;
    UniformSampleGenerator sg = UniformSampleGenerator(1234);
    LightSample sample = {};
    return g_scene.light_system.get_light(LightID(light_id)).sample(ctx, sg, sample) ? uint(sample.flags) : uint(-1);
}

} // namespace light_surface_sampling_tests
