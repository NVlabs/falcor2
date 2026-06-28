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
    public uint active_lobes = uint(LobeTypes::none);

    public __init() { }

    public __init(float3 albedo, float alpha, LobeTypes active_lobes = LobeTypes::none)
    {
        this.albedo = albedo;
        this.alpha = alpha;
        this.active_lobes = uint(active_lobes);
    }

    public bool has_lobe(const LobeTypes lobe_type) { return (active_lobes & uint(lobe_type)) != 0; }

    [Differentiable]
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return float3(0.0);
        if (alpha == 0.0 || !has_lobe(LobeTypes::glossy_reflection))
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
        wo = {};
        pdf = {};
        weight = {};
        lobe_types = LobeTypes::none;

        if (wi.z < BSDF_MIN_COS_THETA)
            return false;

        if (alpha == 0.0) {
            if (!has_lobe(LobeTypes::delta_reflection))
                return false;

            wo = float3(-wi.x, -wi.y, wi.z);
            pdf = 0.0;
            weight = eval_fresnel_schlick(albedo, float3(1.0), wi.z);
            lobe_types = LobeTypes::delta_reflection;
            return true;
        }

        if (!has_lobe(LobeTypes::glossy_reflection))
            return false;

        float half_pdf = {};
        float3 h = sampleGGX_VNDF(alpha, wi, next_float2(sg), half_pdf);
        float wi_dot_h = dot(wi, h);
        if (wi_dot_h <= 0.0)
            return false;

        wo = 2.0 * wi_dot_h * h - wi;
        if (wo.z < BSDF_MIN_COS_THETA)
            return false;

        pdf = half_pdf / (4.0 * wi_dot_h);
        if (pdf <= 0.0)
            return false;

        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
        float g_over_g1_wi = G * (1.0 + evalLambdaGGX(alpha * alpha, wi.z));
        float3 F = eval_fresnel_schlick(albedo, float3(1.0), wi_dot_h);
        weight = F * g_over_g1_wi;
        lobe_types = LobeTypes::glossy_reflection;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return 0.0;
        if (alpha == 0.0 || !has_lobe(LobeTypes::glossy_reflection))
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
    public uint active_lobes = uint(LobeTypes::none);

    public __init() { }

    public __init(float3 transmission_albedo, float alpha, float eta, LobeTypes active_lobes = LobeTypes::none)
    {
        this.transmission_albedo = transmission_albedo;
        this.alpha = alpha;
        this.eta = eta;
        this.active_lobes = uint(active_lobes);
    }

    public bool has_lobe(const LobeTypes lobe_type) { return (active_lobes & uint(lobe_type)) != 0; }

    [Differentiable]
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (min(wi.z, abs(wo.z)) < BSDF_MIN_COS_THETA)
            return float3(0.0);
        if (alpha == 0.0)
            return float3(0.0);

        bool is_reflection = wo.z > 0.0;
        bool has_reflection = has_lobe(LobeTypes::glossy_reflection);
        bool has_transmission = has_lobe(LobeTypes::glossy_transmission);
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
        wo = {};
        pdf = {};
        weight = {};
        lobe_types = LobeTypes::none;

        if (wi.z < BSDF_MIN_COS_THETA)
            return false;

        float lobe_sample = next_float(sg);

        if (alpha == 0.0) {
            bool has_reflection = has_lobe(LobeTypes::delta_reflection);
            bool has_transmission = has_lobe(LobeTypes::delta_transmission);
            if (!(has_reflection || has_transmission))
                return false;

            float cos_theta_t;
            float F = eval_fresnel_dielectric(eta, wi.z, cos_theta_t);

            bool is_reflection = has_reflection;
            if (has_reflection && has_transmission)
                is_reflection = lobe_sample < F;
            else if (has_transmission && F == 1.0)
                return false;

            wo = is_reflection ? float3(-wi.x, -wi.y, wi.z) : float3(-wi.x * eta, -wi.y * eta, -cos_theta_t);
            if (abs(wo.z) < BSDF_MIN_COS_THETA || ((wo.z > 0.0) != is_reflection))
                return false;

            pdf = 0.0;
            weight = is_reflection ? float3(1.0) : transmission_albedo;
            if (!(has_reflection && has_transmission))
                weight *= is_reflection ? F : (1.0 - F);
            lobe_types = is_reflection ? LobeTypes::delta_reflection : LobeTypes::delta_transmission;
            return true;
        }

        bool has_reflection = has_lobe(LobeTypes::glossy_reflection);
        bool has_transmission = has_lobe(LobeTypes::glossy_transmission);
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

        wo = is_reflection ? (2.0 * wi_dot_h * h - wi) : ((eta * wi_dot_h - cos_theta_t) * h - eta * wi);
        if (abs(wo.z) < BSDF_MIN_COS_THETA || ((wo.z > 0.0) != is_reflection))
            return false;

        float wo_dot_h = dot(wo, h);
        if (is_reflection && wo_dot_h <= 0.0)
            return false;
        if (!is_reflection && wo_dot_h >= 0.0)
            return false;

        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, abs(wo.z));
        weight = float3(G * (1.0 + evalLambdaGGX(alpha * alpha, wi.z)));

        if (is_reflection) {
            pdf = half_pdf / (4.0 * wo_dot_h);
        } else {
            float sqrt_denom = wo_dot_h + eta * wi_dot_h;
            float denom = sqrt_denom * sqrt_denom;
            if (denom <= 0.0)
                return false;
            pdf = half_pdf * abs(wo_dot_h) / denom;
            weight *= transmission_albedo * eta * eta;
        }

        if (has_reflection && has_transmission)
            pdf *= is_reflection ? F : (1.0 - F);
        else
            weight *= is_reflection ? F : (1.0 - F);

        if (pdf <= 0.0)
            return false;

        lobe_types = is_reflection ? LobeTypes::glossy_reflection : LobeTypes::glossy_transmission;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        if (min(wi.z, abs(wo.z)) < BSDF_MIN_COS_THETA)
            return 0.0;
        if (alpha == 0.0)
            return 0.0;

        bool is_reflection = wo.z > 0.0;
        bool has_reflection = has_lobe(LobeTypes::glossy_reflection);
        bool has_transmission = has_lobe(LobeTypes::glossy_transmission);
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
