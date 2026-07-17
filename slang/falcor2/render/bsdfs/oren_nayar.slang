// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Oren-Nayar reflection model "Generalization of Lambert's Reflectance Model" for retroreflective surfaces
/// The model will become Lambertian for roughness=0.0
public struct OrenNayarBRDF : IBSDF
{
    /// Diffuse albedo.
    public float3 albedo = {};
    /// Roughness.
    public float roughness = {};

    public __init() { }

    public __init(float3 albedo, float roughness)
    {
        this.albedo = albedo;
        this.roughness = roughness;
    }

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return float3(0.0);

        const float sin_theta_i = sqrt(1.0 - wi.z * wi.z);
        const float sin_phi_i = (sin_theta_i == 0.0 ? 0.0 : clamp(wi.y / sin_theta_i, -1.0, 1.0));
        const float cos_phi_i = (sin_theta_i == 0.0 ? 0.0 : clamp(wi.x / sin_theta_i, -1.0, 1.0));

        float sin_theta_o = sqrt(1.0 - wo.z * wo.z);
        float sin_phi_o = (sin_theta_o == 0.0 ? 0.0 : clamp(wo.y / sin_theta_o, -1.0, 1.0));
        float cos_phi_o = (sin_theta_o == 0.0 ? 0.0 : clamp(wo.x / sin_theta_o, -1.0, 1.0));

        const float cos_diff = max(cos_phi_i * cos_phi_o + sin_phi_i * sin_phi_o, 0.0);

        const float sigma2 = roughness * roughness;
        const float A = 1.0 - 0.5f * (sigma2 / (sigma2 + 0.33f));
        const float B = 0.45f * sigma2 / (sigma2 + 0.09f);

        float sin_alpha, tan_beta;

        if (abs(wi.z) > abs(wo.z)) {
            sin_alpha = sin_theta_o;
            tan_beta = sin_theta_i / abs(wi.z);
        } else {
            sin_alpha = sin_theta_i;
            tan_beta = sin_theta_o / abs(wo.z);
        }

        return albedo * float.INV_PI * (A + B * cos_diff * sin_alpha * tan_beta) * wo.z;
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        // Note that cosine sampling can become rather inefficient if we have high retro-reflectivity
        sample.wo = square_to_cosine_hemisphere_concentric(next_float2(sg));
        sample.pdf = square_to_cosine_hemisphere_concentric_pdf(sample.wo);
        sample.flags = BSDFFlags::diffuse_reflection;

        if (min(wi.z, sample.wo.z) < BSDF_MIN_COS_THETA)
            return false;

        sample.weight = eval(wi, sample.wo, sg, bc) / eval_pdf(wi, sample.wo, bc);
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
            return float3(1.0);
        }
    }
};
