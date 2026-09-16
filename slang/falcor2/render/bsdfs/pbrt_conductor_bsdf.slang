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
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    [Differentiable]
    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (D.is_singular() || min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return BSDFEval();

        float3 h = normalize(wi + wo);
        float wi_dot_h = abs(dot(wi, h));
        float3 F = eval_fresnel_conductor(eta, k, wi_dot_h);
        float pdf = D.eval_pdf(wi, h) / (4.0 * wi_dot_h);
        float3 result = D.G2(wo, wi) * D.eval_ndf(h) * F / (4.0 * abs(wi.z));
        return BSDFEval(result, pdf);
    }

    public bool sample<S : ISampleGenerator>(float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        if (wi.z < BSDF_MIN_COS_THETA)
            return false;

        if (D.is_singular()) {
            sample.wo = float3(-wi.x, -wi.y, wi.z);
            sample.weight = eval_fresnel_conductor(eta, k, wi.z);
            sample.flags = BSDFFlags::delta_reflection;
            sample.pdf = 0.0;
        } else {
            float3 h = D.sample(wi, sg);

            sample.wo = reflect(-wi, h);
            if (sample.wo.z < BSDF_MIN_COS_THETA)
                return false;
            float3 F = eval_fresnel_conductor(eta, k, dot(h, wi));

            sample.weight = F * D.G2(sample.wo, wi) / D.G1(wi);
            sample.flags = BSDFFlags::glossy_reflection;
            sample.pdf = D.eval_pdf(wi, h) / (4.0 * abs(dot(wi, h)));
        }

        return true;
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
