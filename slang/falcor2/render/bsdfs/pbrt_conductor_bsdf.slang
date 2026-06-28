// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct PBRTConductorBSDF : IBSDF, IDifferentiable
{
    public AnisotropicGGX D = {};
    public float3 eta = {};
    public float3 k = {};

    public __init() { }

    public __init(AnisotropicGGX D, float3 eta, float3 k)
    {
        this.D = D;
        this.eta = eta;
        this.k = k;
    }

    [Differentiable]
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (D.is_singular() || min(wi.z, wo.z) < 0.0)
            return float3(0.0);

        float3 h = normalize(wi + wo);
        float wi_dot_h = abs(dot(wi, h));
        float3 F = eval_fresnel_conductor(eta, k, wi_dot_h);
        float3 result = D.G2(wo, wi) * D.eval_ndf(h) * F / (4.0 * abs(wi.z));
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

        if (wi.z < 0.0)
            return false;

        if (D.is_singular()) {
            wo = float3(-wi.x, -wi.y, wi.z);
            weight = eval_fresnel_conductor(eta, k, wi.z);
            lobe_types = LobeTypes::delta_reflection;
            pdf = 1.0;
        } else {
            float3 h = D.sample(wi, sg);

            wo = reflect(-wi, h);
            if (wo.z < 0.0)
                return false;
            float3 F = eval_fresnel_conductor(eta, k, dot(h, wi));

            weight = F * D.G2(wo, wi) / D.G1(wi);
            lobe_types = LobeTypes::glossy_reflection;
            pdf = D.eval_pdf(wi, h) / (4.0 * abs(dot(wi, h)));
        }

        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        if (D.is_singular() || min(wi.z, wo.z) < 0.0)
            return 0.0;
        float3 h = normalize(wi + wo);
        float result = D.eval_pdf(wi, h) / (4.0 * abs(dot(wi, h)));
        return result;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float3 r = eval_fresnel_conductor(eta, k, wi.z);
        return AlbedoContributions(r, 1.0 - r, 0.0, 0.0);
    }

    public RoughnessInformation eval_roughness(const float3 wi) { return RoughnessInformation(D.alpha, float3(0.0)); }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(1.0);
        }
    }
};
