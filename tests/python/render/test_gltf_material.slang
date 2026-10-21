// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

public float2 transform_probe(GLTFMaterial material, float2 uv, uint binding)
{
    return gltf_material::transform_uv(
        uv,
        binding,
        material.transform_mask,
        material.texture_transform_data,
        material.transform_offset
    );
}

public struct GLTFSampleProbe
{
    public bool valid;
    public bool delta;
    public float3 wo;
    public float3 weight;
    public float pdf;
    public float evaluated_pdf;
};

// Exercise real material preparation, including packed factors and exterior IOR.
public float3 analytic_material_eval(uint material_id, float3 wi, float3 wo, float exterior_ior, bool front_facing)
{
    SurfaceInteraction si = {};
    si.normal_ws = float3(0.0, 0.0, 1.0);
    // Back-face directions must lie below the authored geometric normal.
    si.wi_ws = front_facing ? wi : -wi;
    si.front_facing = front_facing;
    si.exterior_ior = exterior_ior;
    si.uv = float2(0.5);
    si.shading_frame_ws = Frame(si.normal_ws);
    let material = g_scene.get_material(MaterialID(material_id));
    let instance = material.setup_material_instance(si, ExplicitLodSampler(0.0));
    UniformSampleGenerator sg = UniformSampleGenerator(1);
    return instance.eval(si, front_facing ? wo : -wo, sg);
}

public GLTFSampleProbe sample_probe(float3 wi, float3 base_color, float metallic, float roughness, uint seed)
{
    GLTFBSDFData data = {};
    data.base_color = base_color;
    data.metallic = metallic;
    data.roughness = roughness;
    GLTFBSDF bsdf = GLTFBSDF(wi, data);
    UniformSampleGenerator sg = UniformSampleGenerator(seed);
    BSDFSample sample;
    GLTFSampleProbe result = {};
    result.valid = bsdf.sample(wi, sample, sg, BSDFContext());
    result.delta = sample.has_flag(BSDFFlags::delta);
    result.wo = sample.wo;
    result.weight = sample.weight;
    result.pdf = sample.pdf;
    result.evaluated_pdf = bsdf.eval_with_pdf(wi, sample.wo, sg, BSDFContext()).pdf;
    return result;
}

[Differentiable]
float3 evaluate_prepared(GLTFBSDF bsdf, no_diff float3 wi, no_diff float3 wo)
{
    UniformSampleGenerator sg = UniformSampleGenerator(1);
    return bsdf.eval(wi, wo, sg, BSDFContext());
}

public float3 diffuse_gradient(float3 wi, float3 wo)
{
    GLTFBSDFData data = {};
    data.base_color = float3(0.5);
    data.roughness = 0.6;
    GLTFBSDF bsdf = GLTFBSDF(wi, data);
    GLTFBSDF.Differential tangent = {};
    tangent.diffuse.albedo = float3(1.0);
    return fwd_diff(evaluate_prepared)(diffPair(bsdf, tangent), wi, wo).d;
}
