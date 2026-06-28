// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Thin (architectural) dielectric to simulate plane parallel materials on a single polygon.
/// Relfection is classic microfacet theory, transmission roughness is calculated from the reflection
/// roughness and the IOR when creating the BSDF. The Frensel is standard glass theory (perfectly specular)
/// and should be replaced by a proper formula
public struct DielectricPlateBSDF : IBSDF
{
    /// Reflection roughness.
    public AnisotropicGGX D = {};
    /// Tranmission roughness.
    public AnisotropicGGX D_inside = {};
    /// Index of refraction.
    public float eta = {};

    public __init() { }

    public __init(AnisotropicGGX D, AnisotropicGGX D_inside, float eta)
    {
        this.D = D;
        this.D_inside = D_inside;
        this.eta = eta;
    }

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (D.is_singular() || eta == 1.0 || wi.z * wo.z == 0.0)
            return float3(0.0);

        bool reflect = wi.z * wo.z > 0.0;
        float rel_eta = eta;
        const float eta_refract = 1.005;
        float3 h = reflect ? normalize(wi + wo) : normalize(wi / eta_refract + wo);
        h *= sign(h.z);
        float wi_dot_h = dot(wi, h);
        float wo_dot_h = dot(wo, h);
        if (wi_dot_h * wi.z < 0.0 || wo_dot_h * wo.z < 0.0)
            return float3(0.0);

        float sin_theta_t_sq = eta * eta * (1 - wi_dot_h * wi_dot_h);
        float cos_theta_t = sqrt(1.0 - sin_theta_t_sq); // No clamp needed

        float r1 = eval_fresnel_dielectric(eta, wi_dot_h);
        float r2 = eval_fresnel_dielectric(eta, -cos_theta_t);
        float t1 = 1.0 - r2;
        float F = r1 + t1 * t1 * r2 / (1.0 - r1 * r2);

        return reflect ? D.G2(wo, wi) * D.eval_ndf(h) * F / (4.0 * abs(wi.z))
                       : D_inside.G2(wo, wi) * D_inside.eval_ndf(h) * (1.0 - F)
                * abs(wi_dot_h * wo_dot_h / (sqr(wo_dot_h + wi_dot_h / eta_refract) * abs(wi.z)));
    }

    public bool sample<S : ISampleGenerator>(
        float3 wi,
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
        lobe_types = {};

        float rel_eta = eta;
        const float eta_refract = 1.005;

        if (eta == 1.0 || D.is_singular()) {
            float sin_theta_t_sq = eta * eta * (1 - wi.z * wi.z);
            float cos_theta_t = sqrt(1 - sin_theta_t_sq); // No clamp needed

            float r1 = eval_fresnel_dielectric(eta, wi.z);
            float r2 = eval_fresnel_dielectric(eta, -cos_theta_t);
            float t1 = 1.0 - r2;
            float F = r1 + t1 * t1 * r2 / (1.0 - r1 * r2);

            if (next_float(sg) < F) {
                wo = float3(-wi.x, -wi.y, wi.z);
                weight = float3(1.0);
                lobe_types = LobeTypes::delta_reflection;
                pdf = 1.0;
            } else {
                wo = refract(-wi, float3(0.0, 0.0, sign(wi.z)), 1.0 / eta_refract);
                weight = float3(1.0);
                lobe_types = LobeTypes::delta_transmission;
                pdf = 1.0;
            }
        } else {
            float3 h = D.sample(wi, sg);
            float wi_dot_h = dot(h, wi);

            float sin_theta_t_sq = eta * eta * (1 - wi_dot_h * wi_dot_h);
            float cos_theta_t = sqrt(1 - sin_theta_t_sq); // No clamp needed

            float r1 = eval_fresnel_dielectric(eta, wi_dot_h);
            float r2 = eval_fresnel_dielectric(eta, -cos_theta_t);
            float t1 = 1.0 - r2;
            float F = r1 + t1 * t1 * r2 / (1.0 - r1 * r2);

            if (next_float(sg) < F) {
                wo = reflect(-wi, h);
                if (wi.z * wo.z < 0.0)
                    return false;

                weight = D.G2(wo, wi) / D.G1(wi);
                lobe_types = LobeTypes::glossy_reflection;
                pdf = F * D.eval_pdf(wi, h) / (4.0 * abs(wi_dot_h));
            } else {
                float3 h = D_inside.sample(wi, sg);
                float wi_dot_h = dot(h, wi);

                wo = refract(-wi, sign(wi.z) * h, 1.0 / eta_refract);
                float wo_dot_h = dot(h, wo);
                if (wi.z * wo.z > 0.0)
                    return false;

                weight = D_inside.G2(wo, wi) / D_inside.G1(wi);
                lobe_types = LobeTypes::glossy_transmission;
                pdf = (1.0 - F) * D_inside.eval_pdf(wi, h) * abs(wo_dot_h) / sqr(wo_dot_h + wi_dot_h / eta_refract);
            }
        }
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        if (D.is_singular() || eta == 1.0 || wi.z * wo.z == 0.0)
            return 0.0;

        bool reflect = wi.z * wo.z > 0.0;
        float rel_eta = eta;
        const float eta_refract = 1.005f;
        float3 h = reflect ? normalize(wi + wo) : normalize(wi / eta_refract + wo);
        h *= sign(h.z);
        float wi_dot_h = dot(wi, h);
        float wo_dot_h = dot(wo, h);
        if (wi_dot_h * wi.z < 0.0 || wo_dot_h * wo.z < 0.0)
            return 0.0;

        float sin_theta_t_sq = eta * eta * (1 - wi_dot_h * wi_dot_h);
        float cos_theta_t = sqrt(1 - sin_theta_t_sq); // No clamp needed

        float r1 = eval_fresnel_dielectric(eta, wi_dot_h);
        float r2 = eval_fresnel_dielectric(eta, -cos_theta_t);
        float t1 = 1.0 - r2;
        float F = r1 + t1 * t1 * r2 / (1.0 - r1 * r2);

        return reflect ? F * D.eval_pdf(wi, h) / (4.0 * abs(wi_dot_h))
                       : (1.0 - F) * D_inside.eval_pdf(wi, h) * abs(wo_dot_h) / sqr(wo_dot_h + wi_dot_h / eta_refract);
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float sin_theta_t_sq = eta * eta * (1 - wi.z * wi.z);
        float cos_theta_t = sqrt(1 - sin_theta_t_sq); // No clamp needed

        float r1 = eval_fresnel_dielectric(eta, wi.z);
        float r2 = eval_fresnel_dielectric(eta, -cos_theta_t);
        float t1 = 1.0 - r2;
        float F = r1 + t1 * t1 * r2 / (1.0 - r1 * r2);

        return AlbedoContributions(F, 0.f, 1.0 - F, 0.f);
    }

    public RoughnessInformation eval_roughness(const float3 wi) { return RoughnessInformation(D.alpha, float3(0.0)); }

    public property float3 ior_as_reflectance
    {
        get {
            const float rsqrt = (1.0 - eta) / (1.0 + eta);
            return rsqrt * rsqrt;
        }
    }
};
