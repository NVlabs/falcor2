// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct PBRTDielectricBSDF : IBSDF
{
    public AnisotropicGGX D = {};
    public float eta = {};

    public __init() { }

    public __init(AnisotropicGGX D, float eta)
    {
        this.D = D;
        this.eta = eta;
    }

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        const float rel_eta = bc.ior_t / bc.ior_i;

        if (D.is_singular() || rel_eta == 1.0 || wi.z * wo.z == 0.0)
            return float3(0.0);

        bool reflect = wi.z * wo.z > 0.0;
        const float rel_eta_inverse = 1.0 / rel_eta;
        float3 h = reflect ? normalize(wi + wo) : normalize(wi / rel_eta + wo);
        h *= sign(h.z);
        float wi_dot_h = dot(wi, h);
        float wo_dot_h = dot(wo, h);
        if (wi_dot_h * wi.z < 0.0 || wo_dot_h * wo.z < 0.0)
            return float3(0.0);
        float F = eval_fresnel_dielectric(rel_eta_inverse, abs(wi_dot_h));
        float3 result = reflect ? D.G2(wo, wi) * D.eval_ndf(h) * F / (4.0 * abs(wi.z))
                                : D.G2(wo, wi) * D.eval_ndf(h) * (1.0 - F)
                * abs(wi_dot_h * wo_dot_h / (sqr(wo_dot_h + wi_dot_h / rel_eta) * abs(wi.z)));
        return result;
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

        const float rel_eta = bc.ior_t / bc.ior_i;
        const float rel_eta_inverse = 1.0 / rel_eta;
        const bool allow_reflection = is_set(bc.sample_lobe_types_hint, LobeTypes::reflection);
        const bool allow_transmission = is_set(bc.sample_lobe_types_hint, LobeTypes::transmission);
        const bool allow_both = allow_reflection && allow_transmission;
        if (!allow_reflection && !allow_transmission)
            return false;

        if (rel_eta == 1 || D.is_singular()) {
            float F = eval_fresnel_dielectric(rel_eta_inverse, abs(wi.z));

            if (allow_reflection && (!allow_transmission || next_float(sg) < F)) {
                wo = float3(-wi.x, -wi.y, wi.z);
                weight = float3(allow_both ? 1.0 : F);
                lobe_types = LobeTypes::delta_reflection;
                pdf = 1.0;
            } else {
                wo = refract(-wi, float3(0.0, 0.0, sign(wi.z)), rel_eta_inverse);
                weight = float3(allow_both ? 1.0 : 1.0 - F);
                lobe_types = LobeTypes::delta_transmission;
                pdf = 1.0;
            }
        } else {
            float3 h = D.sample(wi, sg);
            float wi_dot_h = dot(h, wi);
            float F = eval_fresnel_dielectric(rel_eta_inverse, abs(wi_dot_h));

            if (allow_reflection && (!allow_transmission || next_float(sg) < F)) {
                wo = reflect(-wi, h);
                if (wi.z * wo.z < 0.0)
                    return false;

                weight = D.G2(wo, wi) / D.G1(wi);
                if (!allow_both)
                    weight *= F;
                lobe_types = LobeTypes::glossy_reflection;
            } else {
                wo = refract(-wi, sign(wi.z) * h, rel_eta_inverse);
                if (wi.z * wo.z > 0.0)
                    return false;

                weight = D.G2(wo, wi) / D.G1(wi);
                if (!allow_both)
                    weight *= (1.0 - F);
                lobe_types = LobeTypes::glossy_transmission;
            }
            // Keep the returned sample PDF coherent with eval_pdf() for the
            // emitted direction. Near-singular rough dielectrics are
            // numerically sensitive enough that the PDF computed from the
            // sampled microfacet can drift from the PDF reconstructed from
            // wi/wo, even though both are analytically equivalent.
            pdf = eval_pdf(wi, wo, bc);
            if (pdf <= 0.0)
                return false;
        }
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        const float rel_eta = bc.ior_t / bc.ior_i;
        const float rel_eta_inverse = 1.0 / rel_eta;
        const bool allow_reflection = is_set(bc.sample_lobe_types_hint, LobeTypes::reflection);
        const bool allow_transmission = is_set(bc.sample_lobe_types_hint, LobeTypes::transmission);
        const bool allow_both = allow_reflection && allow_transmission;
        if (D.is_singular() || rel_eta == 1.0 || wi.z * wo.z == 0.0)
            return 0.0;

        bool reflect = wi.z * wo.z > 0.0;
        if ((reflect && !allow_reflection) || (!reflect && !allow_transmission))
            return 0.0;
        float3 h = reflect ? normalize(wi + wo) : normalize(wi / rel_eta + wo);
        h *= sign(h.z);
        float wi_dot_h = dot(wi, h);
        float wo_dot_h = dot(wo, h);
        if (wi_dot_h * wi.z < 0.0 || wo_dot_h * wo.z < 0.0)
            return 0.0;
        float F = eval_fresnel_dielectric(rel_eta_inverse, abs(wi_dot_h));
        float result = reflect
            ? (allow_both ? F : 1.0) * D.eval_pdf(wi, h) / (4.0 * abs(wi_dot_h))
            : (allow_both ? 1.0 - F : 1.0) * D.eval_pdf(wi, h) * abs(wo_dot_h) / sqr(wo_dot_h + wi_dot_h / rel_eta);
        return result;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float3 r = eval_fresnel_dielectric(eta, abs(wi.z));
        return AlbedoContributions(r, 0.0, 1.0 - r, 0.0);
    }

    public RoughnessInformation eval_roughness(const float3 wi) { return RoughnessInformation(D.alpha, float3(0.0)); }

    public property float3 ior_as_reflectance
    {
        get {
            // This is the absolute IOR. At normal incidence we are side invariant.
            const float rsqrt = (1.0 - eta) / (1.0 + eta);
            return rsqrt * rsqrt;
        }
    }
};
