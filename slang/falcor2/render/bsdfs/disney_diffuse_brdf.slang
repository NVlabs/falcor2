// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Disney's diffuse reflection.
/// Based on https ://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
public struct DisneyDiffuseBRDF : IBSDF, IDifferentiable
{
    /// Diffuse albedo.
    public float3 albedo = {};
    /// Roughness before remapping.
    public float roughness = {};

    public __init() { }

    public __init(float3 albedo, float roughness)
    {
        this.albedo = albedo;
        this.roughness = roughness;
    }


    [Differentiable]
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    [Differentiable]
    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return BSDFEval();

        const float pdf = float.INV_PI * wo.z;
        return BSDFEval(eval_weight(wi, wo) * pdf, pdf);
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        sample.wo = square_to_cosine_hemisphere_concentric(next_float2(sg));
        sample.pdf = square_to_cosine_hemisphere_concentric_pdf(sample.wo);
        sample.flags = BSDFFlags::diffuse_reflection;

        if (min(wi.z, sample.wo.z) < BSDF_MIN_COS_THETA)
            return false;

        sample.weight = eval_weight(wi, sample.wo);
        return true;
    }


    public AlbedoContributions eval_albedo(const float3 wi)
    {
        return AlbedoContributions(albedo, 1.0 - albedo, 0.0, 0.0);
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(0.5), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(1.0);
        }
    }

    // Returns f(wi, wo) * pi.
    [Differentiable]
    private float3 eval_weight(const float3 wi, const float3 wo)
    {
        float3 h = normalize(wi + wo);
        float wo_dot_h = dot(wo, h);
        float fd90 = 0.5 + 2.0 * wo_dot_h * wo_dot_h * roughness;
        float fd0 = 1.0;
        float wi_scatter = eval_fresnel_schlick(fd0, fd90, wi.z);
        float wo_scatter = eval_fresnel_schlick(fd0, fd90, wo.z);
        return albedo * wi_scatter * wo_scatter;
    }
};
