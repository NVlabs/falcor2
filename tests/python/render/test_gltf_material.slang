// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

public float2 transform_probe(
    uint material_id,
    float2 uv,
    uint binding,
    uint feature_word,
    uint transform_word,
    uint extra_pointer_word
)
{
    // The caller derives these word indices from the reflected material layout.
    const MaterialData material = g_scene.material_system.get_material_data(MaterialID(material_id));
    const uint feature_mask = material.payload.data[feature_word];
    const uint transform_mask = material.payload.data[transform_word];
    const uint bit = 1u << binding;
    if ((transform_mask & bit) == 0u)
        return uv;
    const BufferPtr extra_data = BufferPtr<void>(
        uint64_t(material.payload.data[extra_pointer_word])
        | (uint64_t(material.payload.data[extra_pointer_word + 1u]) << 32u)
    );
    const uint location = gltf_material::transform_record_offset(feature_mask, transform_mask, bit);
    const GLTFTextureUVTransform transform = extra_data.load<GLTFTextureUVTransform>(location);
    const float3 p = float3(uv, 1.0);
    return float2(dot(transform.row_u, p), dot(transform.row_v, p));
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
    return instance.eval(si, front_facing ? wo : -wo);
}

public GLTFSampleProbe sample_probe(float3 wi, float3 base_color, float metallic, float roughness, uint seed)
{
    GLTFBSDFData<0u> data = {};
    data.base_color = base_color;
    data.metallic = metallic;
    data.roughness = roughness;
    GLTFBSDF<0u> bsdf = GLTFBSDF<0u>(wi, data, 0u);
    UniformSampleGenerator sg = UniformSampleGenerator(seed);
    BSDFSample sample;
    GLTFSampleProbe result = {};
    result.valid = bsdf.sample(wi, make_bsdf_sample_input(sg), BSDFContext(), sample);
    result.delta = sample.has_flag(BSDFFlags::delta);
    result.wo = sample.wo;
    result.weight = sample.weight;
    result.pdf = sample.pdf;
    result.evaluated_pdf = bsdf.eval_with_pdf(wi, sample.wo, BSDFContext()).pdf;
    return result;
}

public struct GLTFClearcoatSampleProbe
{
    public bool valid;
    public bool sampled_coat;
    public bool delta;
    public float pdf;
    public float evaluated_pdf;
    public float3 weighted_value;
    public float3 evaluated_value;
};

// Exercise the layered material instance rather than just the base GLTFBSDF.
public GLTFClearcoatSampleProbe
clearcoat_instance_sample_probe(float3 wi, float3 base_color, float metallic, float roughness, uint seed)
{
    SurfaceInteraction si = {};
    si.normal_ws = float3(0.0, 0.0, 1.0);
    si.wi_ws = wi;
    si.front_facing = true;
    si.exterior_ior = 1.0;
    si.shading_frame_ws = Frame(si.normal_ws);

    GLTFMaterialInstance<gltfmat::FeatureBits::CLEARCOAT> instance = {};
    instance.feature_mask = gltfmat::FeatureBits::CLEARCOAT;
    instance.shading_frame_ws = si.shading_frame_ws;
    instance.bsdf_data.base_color = base_color;
    instance.bsdf_data.metallic = metallic;
    instance.bsdf_data.roughness = roughness;
    instance.clearcoat.set(GLTFClearcoatState(si.shading_frame_ws, 1.0, 0.25));

    UniformSampleGenerator sg = UniformSampleGenerator(seed);
    const MaterialSampleInput input = make_material_sample_input(sg);
    GLTFClearcoatSampleProbe result = {};
    result.sampled_coat = input.u_lobe < gltf_material::clearcoat_weight(1.0, wi.z);
    MaterialSample sample = {};
    result.valid = instance.sample(si, input, sample);
    if (result.valid) {
        result.delta = (sample.flags & BSDFFlags::delta) != BSDFFlags::none;
        result.pdf = sample.pdf;
        result.weighted_value = sample.weight * sample.pdf;
        const BSDFEval evaluated = instance.eval_with_pdf(si, sample.wo_ws);
        result.evaluated_pdf = evaluated.pdf;
        result.evaluated_value = evaluated.value;
    }
    return result;
}

// Check that compiling all optional branches but disabling them at runtime
// preserves the core BSDF's forward result and PDF.
public float4 compare_core_exact_flexible(float3 wi, float3 wo, float3 base_color, float metallic, float roughness)
{
    GLTFBSDFData<0u> exact_data = {};
    exact_data.base_color = base_color;
    exact_data.metallic = metallic;
    exact_data.roughness = roughness;
    GLTFBSDFData<gltfmat::FeatureBits::ALL_MR> flexible_data = {};
    flexible_data.base_color = base_color;
    flexible_data.metallic = metallic;
    flexible_data.roughness = roughness;
    GLTFBSDF<0u> exact = GLTFBSDF<0u>(wi, exact_data, 0u);
    GLTFBSDF<gltfmat::FeatureBits::ALL_MR> flexible = GLTFBSDF<gltfmat::FeatureBits::ALL_MR>(wi, flexible_data, 0u);
    BSDFEval exact_result = exact.eval_with_pdf(wi, wo, BSDFContext());
    BSDFEval flexible_result = flexible.eval_with_pdf(wi, wo, BSDFContext());
    return float4(abs(exact_result.value - flexible_result.value), abs(exact_result.pdf - flexible_result.pdf));
}

public float3 core_bsdf_eval(float3 wi, float3 wo, float3 base_color, float metallic, float roughness)
{
    GLTFBSDFData<0u> data = {};
    data.base_color = base_color;
    data.metallic = metallic;
    data.roughness = roughness;
    GLTFBSDF<0u> bsdf = GLTFBSDF<0u>(wi, data, 0u);
    return bsdf.eval(wi, wo, BSDFContext());
}

// Compare the instance's cheap lobe classification with the fully prepared
// core BSDF. Clearcoat is layered by the instance after classifying the base.
public uint2
core_instance_lobe_types(float3 base_color, float metallic, float roughness, bool coat_enabled, float coat_roughness)
{
    GLTFMaterialInstance<gltfmat::FeatureBits::CLEARCOAT> instance = {};
    instance.bsdf_data.base_color = base_color;
    instance.bsdf_data.metallic = metallic;
    instance.bsdf_data.roughness = roughness;
    instance.shading_frame_ws = Frame(float3(0.0, 0.0, 1.0));
    instance.clearcoat.set(GLTFClearcoatState(instance.shading_frame_ws, 1.0, coat_roughness));
    if (coat_enabled)
        instance.feature_mask = gltfmat::FeatureBits::CLEARCOAT;

    SurfaceInteraction si = {};
    si.wi_ws = float3(0.0, 0.0, 1.0);
    GLTFBSDF<gltfmat::FeatureBits::CLEARCOAT> prepared
        = GLTFBSDF<gltfmat::FeatureBits::CLEARCOAT>(si.wi_ws, instance.bsdf_data, instance.feature_mask);
    return uint2(uint(instance.get_lobe_types(si)), uint(prepared.get_lobe_types()));
}

[Differentiable]
float3 evaluate_specgloss_prepared(GLTFSpecGlossBSDF bsdf, no_diff float3 wi, no_diff float3 wo)
{
    return bsdf.eval(wi, wo, BSDFContext());
}

public float3 specgloss_diffuse_gradient(float3 wi, float3 wo)
{
    GLTFSpecGlossBSDFData data = {};
    data.diffuse = float3(0.5);
    data.specular = float3(0.1, 0.5, 0.3);
    data.glossiness = 0.4;
    GLTFSpecGlossBSDF bsdf = GLTFSpecGlossBSDF(wi, data);
    GLTFSpecGlossBSDF.Differential tangent = {};
    tangent.diffuse.albedo = float3(1.0);
    return fwd_diff(evaluate_specgloss_prepared)(diffPair(bsdf, tangent), wi, wo).d;
}
