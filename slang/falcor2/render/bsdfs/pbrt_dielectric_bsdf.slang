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


    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        const float rel_eta = bc.ior_t / bc.ior_i;
        const float rel_eta_inverse = 1.0 / rel_eta;
        const bool allow_reflection = is_set(bc.sample_lobe_types_hint, BSDFFlags::reflection);
        const bool allow_transmission = is_set(bc.sample_lobe_types_hint, BSDFFlags::transmission);
        const bool allow_both = allow_reflection && allow_transmission;

        if (D.is_singular() || rel_eta == 1.0 || min(wi.z, abs(wo.z)) < BSDF_MIN_COS_THETA)
            return BSDFEval();

        bool reflect = wi.z * wo.z > 0.0;
        float3 h = reflect ? normalize(wi + wo) : normalize(wi / rel_eta + wo);
        h *= sign(h.z);
        float wi_dot_h = dot(wi, h);
        float wo_dot_h = dot(wo, h);
        if (wi_dot_h * wi.z < 0.0 || wo_dot_h * wo.z < 0.0)
            return BSDFEval();
        float F = eval_fresnel_dielectric(rel_eta_inverse, abs(wi_dot_h));
        float3 value = reflect ? D.G2(wo, wi) * D.eval_ndf(h) * F / (4.0 * abs(wi.z))
                               : D.G2(wo, wi) * D.eval_ndf(h) * (1.0 - F)
                * abs(wi_dot_h * wo_dot_h / (sqr(wo_dot_h + wi_dot_h / rel_eta) * abs(wi.z))) * sqr(rel_eta_inverse);
        float pdf = 0.0;
        if ((reflect && allow_reflection) || (!reflect && allow_transmission)) {
            pdf = reflect
                ? (allow_both ? F : 1.0) * D.eval_pdf(wi, h) / (4.0 * abs(wi_dot_h))
                : (allow_both ? 1.0 - F : 1.0) * D.eval_pdf(wi, h) * abs(wo_dot_h) / sqr(wo_dot_h + wi_dot_h / rel_eta);
        }
        return BSDFEval(value, pdf);
    }

    public bool sample<S : ISampleGenerator>(float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        if (wi.z < BSDF_MIN_COS_THETA)
            return false;

        const float rel_eta = bc.ior_t / bc.ior_i;
        const float rel_eta_inverse = 1.0 / rel_eta;
        const bool allow_reflection = is_set(bc.sample_lobe_types_hint, BSDFFlags::reflection);
        const bool allow_transmission = is_set(bc.sample_lobe_types_hint, BSDFFlags::transmission);
        const bool allow_both = allow_reflection && allow_transmission;
        if (!allow_reflection && !allow_transmission)
            return false;

        if (rel_eta == 1 || D.is_singular()) {
            float F = eval_fresnel_dielectric(rel_eta_inverse, abs(wi.z));
            float p_reflection = allow_reflection ? F : 0.0;
            float p_transmission = allow_transmission ? 1.0 - F : 0.0;
            if (p_reflection == 0.0 && p_transmission == 0.0)
                return false;

            if (p_reflection > 0.0
                && (p_transmission == 0.0 || next_float(sg) < p_reflection / (p_reflection + p_transmission))) {
                sample.wo = float3(-wi.x, -wi.y, wi.z);
                sample.weight = float3(allow_both ? 1.0 : F);
                sample.flags = BSDFFlags::delta_reflection;
                sample.pdf = 0.0;
            } else {
                sample.wo = refract(-wi, float3(0.0, 0.0, sign(wi.z)), rel_eta_inverse);
                if (all(sample.wo == 0.0))
                    return false;
                sample.weight = float3((allow_both ? 1.0 : 1.0 - F) * sqr(rel_eta_inverse));
                sample.flags = BSDFFlags::delta_transmission;
                sample.pdf = 0.0;
            }
        } else {
            float3 h = D.sample(wi, sg);
            float wi_dot_h = dot(h, wi);
            float F = eval_fresnel_dielectric(rel_eta_inverse, abs(wi_dot_h));
            float p_reflection = allow_reflection ? F : 0.0;
            float p_transmission = allow_transmission ? 1.0 - F : 0.0;
            if (p_reflection == 0.0 && p_transmission == 0.0)
                return false;

            if (p_reflection > 0.0
                && (p_transmission == 0.0 || next_float(sg) < p_reflection / (p_reflection + p_transmission))) {
                sample.wo = reflect(-wi, h);
                if (sample.wo.z < BSDF_MIN_COS_THETA)
                    return false;

                sample.weight = D.G2(sample.wo, wi) / D.G1(wi);
                if (!allow_both)
                    sample.weight *= F;
                sample.flags = BSDFFlags::glossy_reflection;
            } else {
                sample.wo = refract(-wi, sign(wi.z) * h, rel_eta_inverse);
                if (all(sample.wo == 0.0) || sample.wo.z > -BSDF_MIN_COS_THETA)
                    return false;

                sample.weight = D.G2(sample.wo, wi) / D.G1(wi);
                if (!allow_both)
                    sample.weight *= (1.0 - F);
                sample.weight *= sqr(rel_eta_inverse);
                sample.flags = BSDFFlags::glossy_transmission;
            }
            // Keep the returned sample PDF coherent with eval_pdf() for the
            // emitted direction. Near-singular rough dielectrics are
            // numerically sensitive enough that the PDF computed from the
            // sampled microfacet can drift from the PDF reconstructed from
            // wi/wo, even though both are analytically equivalent.
            sample.pdf = eval_pdf(wi, sample.wo, bc);
            if (sample.pdf <= 0.0)
                return false;
        }
        return true;
    }


    public AlbedoContributions eval_albedo(const float3 wi)
    {
        // eta is the absolute material IOR; context-free properties assume incidence from air.
        float3 r = eval_fresnel_dielectric(1.0 / eta, abs(wi.z));
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
