// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

static const int TEST_LOBE_COUNT = 5;

public struct MaterialXLobeNormalRepairProbeResult
{
    uint no_hint_bsdf_count = {};
    uint adjusted_bsdf_count = {};
    float3 no_hint_bsdf_normal[TEST_LOBE_COUNT] = {};
    float3 adjusted_bsdf_normal[TEST_LOBE_COUNT] = {};
    float3 adjusted_bsdf_tangent[TEST_LOBE_COUNT] = {};
    float3 eval = {};
    float pdf = {};
    MaterialSample sample = {};
};

SurfaceInteraction make_surface_interaction(float2 uv, float3 wi)
{
    SurfaceInteraction si = {};
    si.position_ws = float3(uv, 0.0);
    si.wi_ws = wi;
    si.uv = uv;
    si.shading_frame_ws = Frame(float3(0.0, 1.0, 0.0));
    si.normal_ws = si.shading_frame_ws.normal;
    si.front_facing = dot(si.wi_ws, si.normal_ws) >= 0.0;
    si.exterior_ior = 1.0;
    return si;
}

public MaterialXLobeNormalRepairProbeResult
materialx_lobe_normal_repair_probe(uint3 seeds, uint32_t material_id, float2 uv, float3 wi, float3 wo)
{
    MaterialXLobeNormalRepairProbeResult result = {};
    const SurfaceInteraction si = make_surface_interaction(uv, wi);
    const ExplicitLodSampler lod_sampler = ExplicitLodSampler(0.0);
    let material = g_scene.get_material(MaterialID(material_id));
    let no_hint_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);
    let adjusted_instance
        = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::adjust_shading_normal);

    if (let properties = no_hint_instance.collect_extra_bsdf_properties(si, wo)) {
        result.no_hint_bsdf_count = properties.bsdf_count;
        [ForceUnroll]
        for (int i = 0; i < TEST_LOBE_COUNT; ++i)
            result.no_hint_bsdf_normal[i] = properties.bsdf_N[i];
    }
    if (let properties = adjusted_instance.collect_extra_bsdf_properties(si, wo)) {
        result.adjusted_bsdf_count = properties.bsdf_count;
        [ForceUnroll]
        for (int i = 0; i < TEST_LOBE_COUNT; ++i) {
            result.adjusted_bsdf_normal[i] = properties.bsdf_N[i];
            result.adjusted_bsdf_tangent[i] = properties.bsdf_T[i];
        }
    }

    var eval_sg = TinyUniformSampleGenerator(seeds.xy, seeds.z);
    result.eval = adjusted_instance.eval(si, wo, eval_sg);
    result.pdf = adjusted_instance.eval_pdf(si, wo);

    var sample_sg = TinyUniformSampleGenerator(seeds.xy, seeds.z);
    if (!adjusted_instance.sample(si, sample_sg, result.sample))
        result.sample.pdf = 0.0;

    return result;
}
