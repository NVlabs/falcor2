// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Lambertian diffuse transmission.
public struct LambertDiffuseBTDF : IBSDF, IDifferentiable
{
    /// Diffuse albedo.
    public float3 albedo = {};

    public __init() { }

    public __init(float3 albedo) { this.albedo = albedo; }


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
        if (min(wi.z, -wo.z) < BSDF_MIN_COS_THETA)
            return BSDFEval();

        const float pdf = float.INV_PI * -wo.z;
        return BSDFEval(albedo * pdf, pdf);
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        sample.wo = square_to_cosine_hemisphere_concentric(next_float2(sg));
        sample.pdf = square_to_cosine_hemisphere_concentric_pdf(sample.wo);
        sample.wo.z = -sample.wo.z;
        sample.flags = BSDFFlags::diffuse_transmission;

        if (min(wi.z, -sample.wo.z) < BSDF_MIN_COS_THETA)
            return false;

        sample.weight = albedo;
        return true;
    }


    public AlbedoContributions eval_albedo(const float3 wi)
    {
        return AlbedoContributions(0.0, 0.0, albedo, 1.0 - albedo);
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(0.5), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }
}
