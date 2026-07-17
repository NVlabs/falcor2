// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

///     Generalized Schlick microfacet BSDF.
///
/// Implementation references:
/// - MaterialX generalized schlick node wiring:
///   external/MaterialX/libraries/pbrlib/genosl/mx_generalized_schlick_bsdf.osl
/// - MaterialX generalized schlick Fresnel definition:
///   external/MaterialX/libraries/pbrlib/genglsl/lib/mx_microfacet.glsl
/// - Microfacet reflection/transmission formulation adapted from:
///   slang/falcor2/render/bsdfs/pbrt_dielectric_bsdf.slang
public struct GeneralizedSchlickBSDF : IBSDF
{
    public AnisotropicGGX D = {};
    public float3 color0 = {};
    public float3 color90 = {};
    public float exponent = {};

    public __init() { }

    public __init(AnisotropicGGX D, float3 color0, float3 color90, float exponent)
    {
        this.D = D;
        this.color0 = color0;
        this.color90 = color90;
        this.exponent = exponent;
    }

    float3 eval_fresnel(float cos_theta)
    {
        return eval_fresnel_generalized_schlick(color0, color90, exponent, cos_theta);
    }

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (D.is_singular() || wi.z * wo.z <= 0.0)
            return float3(0.0);

        float3 h = normalize(wi + wo);
        h *= sign(h.z);
        float wi_dot_h = dot(wi, h);
        float wo_dot_h = dot(wo, h);
        if (wi_dot_h * wi.z < 0.0 || wo_dot_h * wo.z < 0.0)
            return float3(0.0);

        float3 F = eval_fresnel(abs(wi_dot_h));
        return D.G2(wo, wi) * D.eval_ndf(h) * F / (4.0 * abs(wi.z));
    }

    public bool sample<S : ISampleGenerator>(float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        if (wi.z < 0.0)
            return false;

        if (D.is_singular()) {
            sample.wo = float3(-wi.x, -wi.y, wi.z);
            sample.weight = eval_fresnel(wi.z);
            sample.flags = BSDFFlags::delta_reflection;
            sample.pdf = 1.0;
        } else {
            float3 h = D.sample(wi, sg);
            sample.wo = reflect(-wi, h);
            if (sample.wo.z < 0.0)
                return false;

            float3 F = eval_fresnel(abs(dot(h, wi)));
            sample.weight = F * D.G2(sample.wo, wi) / D.G1(wi);
            sample.flags = BSDFFlags::glossy_reflection;
            sample.pdf = D.eval_pdf(wi, h) / (4.0 * abs(dot(wi, h)));
        }
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        if (D.is_singular() || wi.z * wo.z <= 0.0)
            return 0.0;
        float3 h = normalize(wi + wo);
        return D.eval_pdf(wi, h) / (4.0 * abs(dot(wi, h)));
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float3 r = eval_fresnel(abs(wi.z));
        return AlbedoContributions(r, 0.0, 1.0 - r, 0.0);
    }

    public RoughnessInformation eval_roughness(const float3 wi) { return RoughnessInformation(D.alpha, float3(0.0)); }

    public property float3 ior_as_reflectance
    {
        get {
            return color0;
        }
    }
};
