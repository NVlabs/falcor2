// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Specular reflection using isotropic GGX microfacets and Schlick Fresnel.
public struct SpecularMicrofacetBRDF : IBSDF, IDifferentiable
{
    /// Specular reflectance at normal incidence.
    public float3 albedo = {};
    /// GGX width parameter.
    public float alpha = {};
    /// Active lobes for sampling and evaluation.
    public uint active_lobes = uint(BSDFFlags::none);

    public __init() { }

    public __init(float3 albedo, float alpha, BSDFFlags active_lobes = BSDFFlags::none)
    {
        this.albedo = albedo;
        this.alpha = alpha;
        this.active_lobes = uint(active_lobes);
    }

    public bool has_lobe(const BSDFFlags lobe_type) { return (active_lobes & uint(lobe_type)) != 0; }

    [Differentiable]
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return float3(0.0);
        if (alpha == 0.0 || !has_lobe(BSDFFlags::glossy_reflection))
            return float3(0.0);

        float3 h = normalize(wi + wo);
        float wi_dot_h = dot(wi, h);
        if (wi_dot_h <= 0.0)
            return float3(0.0);

        float D = evalNdfGGX(alpha, h.z);
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
        float3 F = eval_fresnel_schlick(albedo, float3(1.0), wi_dot_h);
        return F * D * G * 0.25 / wi.z;
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        if (wi.z < BSDF_MIN_COS_THETA)
            return false;

        if (alpha == 0.0) {
            if (!has_lobe(BSDFFlags::delta_reflection))
                return false;

            sample.wo = float3(-wi.x, -wi.y, wi.z);
            sample.pdf = 0.0;
            sample.weight = eval_fresnel_schlick(albedo, float3(1.0), wi.z);
            sample.flags = BSDFFlags::delta_reflection;
            return true;
        }

        if (!has_lobe(BSDFFlags::glossy_reflection))
            return false;

        float half_pdf = {};
        float3 h = sampleGGX_VNDF(alpha, wi, next_float2(sg), half_pdf);
        float wi_dot_h = dot(wi, h);
        if (wi_dot_h <= 0.0)
            return false;

        sample.wo = 2.0 * wi_dot_h * h - wi;
        if (sample.wo.z < BSDF_MIN_COS_THETA)
            return false;

        sample.pdf = half_pdf / (4.0 * wi_dot_h);
        if (sample.pdf <= 0.0)
            return false;

        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, sample.wo.z);
        float g_over_g1_wi = G * (1.0 + evalLambdaGGX(alpha * alpha, wi.z));
        float3 F = eval_fresnel_schlick(albedo, float3(1.0), wi_dot_h);
        sample.weight = F * g_over_g1_wi;
        sample.flags = BSDFFlags::glossy_reflection;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return 0.0;
        if (alpha == 0.0 || !has_lobe(BSDFFlags::glossy_reflection))
            return 0.0;

        float3 h = normalize(wi + wo);
        float wi_dot_h = dot(wi, h);
        if (wi_dot_h <= 0.0)
            return 0.0;

        return evalPdfGGX_VNDF(alpha, wi, h) / (4.0 * wi_dot_h);
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float3 r = eval_fresnel_schlick(albedo, float3(1.0), wi.z);
        return AlbedoContributions(r, float3(1.0) - r, float3(0.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(alpha), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            return albedo;
        }
    }
};

/// Specular reflection and transmission using isotropic GGX microfacets.
///
/// This follows legacy Falcor's eta convention: eta is eta_i / eta_t.
public struct SpecularMicrofacetBSDF : IBSDF, IDifferentiable
{
    /// Transmission tint.
    public float3 transmission_albedo = {};
    /// GGX width parameter.
    public float alpha = {};
    /// Relative index of refraction (eta_i / eta_t).
    public float eta = {};
    /// Active lobes for sampling and evaluation.
    public uint active_lobes = uint(BSDFFlags::none);

    public __init() { }

    public __init(float3 transmission_albedo, float alpha, float eta, BSDFFlags active_lobes = BSDFFlags::none)
    {
        this.transmission_albedo = transmission_albedo;
        this.alpha = alpha;
        this.eta = eta;
        this.active_lobes = uint(active_lobes);
    }

    public bool has_lobe(const BSDFFlags lobe_type) { return (active_lobes & uint(lobe_type)) != 0; }

    [Differentiable]
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (min(wi.z, abs(wo.z)) < BSDF_MIN_COS_THETA)
            return float3(0.0);
        if (alpha == 0.0)
            return float3(0.0);

        bool is_reflection = wo.z > 0.0;
        bool has_reflection = has_lobe(BSDFFlags::glossy_reflection);
        bool has_transmission = has_lobe(BSDFFlags::glossy_transmission);
        if ((is_reflection && !has_reflection) || (!is_reflection && !has_transmission))
            return float3(0.0);

        float3 h = normalize(wo + wi * (is_reflection ? 1.0 : eta));
        h *= sign(h.z);

        float wi_dot_h = dot(wi, h);
        float wo_dot_h = dot(wo, h);
        if (wi_dot_h <= 0.0)
            return float3(0.0);
        if (is_reflection && wo_dot_h <= 0.0)
            return float3(0.0);
        if (!is_reflection && wo_dot_h >= 0.0)
            return float3(0.0);

        float D = evalNdfGGX(alpha, h.z);
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, abs(wo.z));
        float F = eval_fresnel_dielectric(eta, wi_dot_h);

        if (is_reflection)
            return F * D * G * 0.25 / wi.z;

        float sqrt_denom = wo_dot_h + eta * wi_dot_h;
        float denom = sqrt_denom * sqrt_denom;
        if (denom <= 0.0)
            return float3(0.0);

        float t = eta * eta * wi_dot_h * wo_dot_h / (wi.z * denom);
        return transmission_albedo * (1.0 - F) * D * G * abs(t);
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        if (wi.z < BSDF_MIN_COS_THETA)
            return false;

        float lobe_sample = next_float(sg);

        if (alpha == 0.0) {
            bool has_reflection = has_lobe(BSDFFlags::delta_reflection);
            bool has_transmission = has_lobe(BSDFFlags::delta_transmission);
            if (!(has_reflection || has_transmission))
                return false;

            float cos_theta_t;
            float F = eval_fresnel_dielectric(eta, wi.z, cos_theta_t);

            bool is_reflection = has_reflection;
            if (has_reflection && has_transmission)
                is_reflection = lobe_sample < F;
            else if (has_transmission && F == 1.0)
                return false;

            sample.wo = is_reflection ? float3(-wi.x, -wi.y, wi.z) : float3(-wi.x * eta, -wi.y * eta, -cos_theta_t);
            if (abs(sample.wo.z) < BSDF_MIN_COS_THETA || ((sample.wo.z > 0.0) != is_reflection))
                return false;

            sample.pdf = 0.0;
            sample.weight = is_reflection ? float3(1.0) : transmission_albedo;
            if (!(has_reflection && has_transmission))
                sample.weight *= is_reflection ? F : (1.0 - F);
            sample.flags = is_reflection ? BSDFFlags::delta_reflection : BSDFFlags::delta_transmission;
            return true;
        }

        bool has_reflection = has_lobe(BSDFFlags::glossy_reflection);
        bool has_transmission = has_lobe(BSDFFlags::glossy_transmission);
        if (!(has_reflection || has_transmission))
            return false;

        float half_pdf = {};
        float3 h = sampleGGX_VNDF(alpha, wi, next_float2(sg), half_pdf);
        float wi_dot_h = dot(wi, h);
        if (wi_dot_h <= 0.0)
            return false;

        float cos_theta_t;
        float F = eval_fresnel_dielectric(eta, wi_dot_h, cos_theta_t);

        bool is_reflection = has_reflection;
        if (has_reflection && has_transmission)
            is_reflection = lobe_sample < F;
        else if (has_transmission && F == 1.0)
            return false;

        sample.wo = is_reflection ? (2.0 * wi_dot_h * h - wi) : ((eta * wi_dot_h - cos_theta_t) * h - eta * wi);
        if (abs(sample.wo.z) < BSDF_MIN_COS_THETA || ((sample.wo.z > 0.0) != is_reflection))
            return false;

        float wo_dot_h = dot(sample.wo, h);
        if (is_reflection && wo_dot_h <= 0.0)
            return false;
        if (!is_reflection && wo_dot_h >= 0.0)
            return false;

        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, abs(sample.wo.z));
        sample.weight = float3(G * (1.0 + evalLambdaGGX(alpha * alpha, wi.z)));

        if (is_reflection) {
            sample.pdf = half_pdf / (4.0 * wo_dot_h);
        } else {
            float sqrt_denom = wo_dot_h + eta * wi_dot_h;
            float denom = sqrt_denom * sqrt_denom;
            if (denom <= 0.0)
                return false;
            sample.pdf = half_pdf * abs(wo_dot_h) / denom;
            sample.weight *= transmission_albedo * eta * eta;
        }

        if (has_reflection && has_transmission)
            sample.pdf *= is_reflection ? F : (1.0 - F);
        else
            sample.weight *= is_reflection ? F : (1.0 - F);

        if (sample.pdf <= 0.0)
            return false;

        sample.flags = is_reflection ? BSDFFlags::glossy_reflection : BSDFFlags::glossy_transmission;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        if (min(wi.z, abs(wo.z)) < BSDF_MIN_COS_THETA)
            return 0.0;
        if (alpha == 0.0)
            return 0.0;

        bool is_reflection = wo.z > 0.0;
        bool has_reflection = has_lobe(BSDFFlags::glossy_reflection);
        bool has_transmission = has_lobe(BSDFFlags::glossy_transmission);
        if ((is_reflection && !has_reflection) || (!is_reflection && !has_transmission))
            return 0.0;

        float3 h = normalize(wo + wi * (is_reflection ? 1.0 : eta));
        h *= sign(h.z);

        float wi_dot_h = dot(wi, h);
        float wo_dot_h = dot(wo, h);
        if (wi_dot_h <= 0.0)
            return 0.0;
        if (is_reflection && wo_dot_h <= 0.0)
            return 0.0;
        if (!is_reflection && wo_dot_h >= 0.0)
            return 0.0;

        float F = eval_fresnel_dielectric(eta, wi_dot_h);
        float result = evalPdfGGX_VNDF(alpha, wi, h);
        if (is_reflection) {
            result /= 4.0 * wo_dot_h;
        } else {
            float sqrt_denom = wo_dot_h + eta * wi_dot_h;
            float denom = sqrt_denom * sqrt_denom;
            if (denom <= 0.0)
                return 0.0;
            result *= abs(wo_dot_h) / denom;
        }

        if (has_reflection && has_transmission)
            result *= is_reflection ? F : (1.0 - F);

        return result;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float3 r = eval_fresnel_dielectric(eta, wi.z);
        return AlbedoContributions(float3(0.0), float3(0.0), float3(1.0) - r, r);
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(alpha), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            float r_sqrt = (1.0 - eta) / (1.0 + eta);
            return float3(r_sqrt * r_sqrt);
        }
    }
};
