// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Lambertian diffuse reflection.
public struct LambertDiffuseBRDF : IBSDF, IDifferentiable
{
    /// Diffuse albedo.
    public float3 albedo = {};

    public __init() { }

    public __init(float3 albedo) { this.albedo = albedo; }

    [Differentiable]
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return float3(0.0);

        return float.INV_PI * albedo * wo.z;
    }

    public bool sample<S : ISampleGenerator>(
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 weight,
        out LobeTypes lobe_types,
        inout S sg,
        const BSDFContext bc
    )
    {
        wo = square_to_cosine_hemisphere_concentric(next_float2(sg));
        pdf = square_to_cosine_hemisphere_concentric_pdf(wo);
        lobe_types = LobeTypes::diffuse_reflection;

        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            weight = {};
            return false;
        }

        weight = albedo;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return 0.0;

        return float.INV_PI * wo.z;
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
            return float3(0.0);
        }
    }
}
