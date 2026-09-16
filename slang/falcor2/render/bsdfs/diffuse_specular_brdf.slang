// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public static const float MIN_COS_THETA = 1e-6f;

/// Simple diffuse/specular BRDF used for StandardMaterial.
public struct DiffuseSpecularBRDF : IBSDF, IDifferentiable
{
    /// Diffuse albedo.
    public float3 diffuse = {};
    /// Specular albedo.
    public float3 specular = {};
    /// Roughness (linear).
    public float roughness = {};

    [Differentiable]
    public __init() { }

    [Differentiable]
    public __init(float3 albedo, float3 specular, float metallic, float roughness)
    {
        this.diffuse = albedo * (1.0 - metallic);
        this.specular = lerp(float3(specular * 0.08), albedo, metallic);
        // Avoid zero roughness.
        this.roughness = clamp(roughness, 0.05, 1.0);
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
        if (min(wi.z, wo.z) < MIN_COS_THETA)
            return BSDFEval();

        float3 fd = diffuse_disney(wi, wo, diffuse, roughness);
        float3 fs = specular_microfacet(wi, wo, specular, roughness);

        float diffuse_weight = luminance(diffuse);
        float specular_weight = max(0.001, luminance(specular));
        float norm = 1.0 / (diffuse_weight + specular_weight);
        diffuse_weight *= norm;
        specular_weight *= norm;

        float alpha = roughness * roughness;
        float pdf = 0.0;

        if (diffuse_weight > 0.0)
            pdf += diffuse_weight * diffuse_pdf(wo);
        if (specular_weight > 0.0)
            pdf += specular_weight * specular_microfacet_pdf(wi, wo, alpha);

        return BSDFEval((fd + fs) * wo.z, pdf);
    }

    [Differentiable]
    public bool sample<S : ISampleGenerator>(float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        float diffuse_weight = luminance(diffuse);
        float specular_weight = max(0.001, luminance(specular));
        float norm = 1.0 / (diffuse_weight + specular_weight);
        diffuse_weight *= norm;
        specular_weight *= norm;

        float alpha = roughness * roughness;
        float u_select = no_diff next_float(sg);
        float2 u = no_diff next_float2(sg);

        if (u_select < diffuse_weight) {
            sample.wo = square_to_cosine_hemisphere_concentric(u);
            sample.flags = BSDFFlags::diffuse_reflection;
            sample.pdf = square_to_cosine_hemisphere_concentric_pdf(sample.wo);
            sample.pdf = sample.pdf * diffuse_weight + specular_weight * specular_microfacet_pdf(wi, sample.wo, alpha);
        } else {
            if (!specular_microfacet_sample(wi, alpha, u, sample.wo, sample.pdf))
                return false;
            sample.flags = BSDFFlags::glossy_reflection;
            sample.pdf = sample.pdf * specular_weight + diffuse_weight * diffuse_pdf(sample.wo);
        }

        if (min(wi.z, sample.wo.z) < MIN_COS_THETA || sample.pdf == 0.0)
            return false;

        sample.weight = eval(wi, sample.wo, sg, bc) / sample.pdf;
        return true;
    }

    // Slang cannot propagate derivatives through the generic IBSDF default, so keep the differentiable concrete alias.
    [Differentiable]
    public override float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf(wi, wo, sg, bc).pdf;
    }

    // Internal functions

    // Based on https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
    [Differentiable]
    float3 diffuse_disney(const float3 wi, const float3 wo, const float3 albedo, const float roughness)
    {
        float3 h = normalize(wi + wo);
        float wo_dot_h = max(0.0, dot(wo, h));
        float fd90 = 0.5 + 2.0 * wo_dot_h * wo_dot_h * roughness;
        float fd0 = 1.0;
        float wi_scatter = eval_fresnel_schlick(fd0, fd90, wi.z);
        float wo_scatter = eval_fresnel_schlick(fd0, fd90, wo.z);
        return albedo * (wi_scatter * wo_scatter * float.INV_PI);
    }

    // MaterialX 1.39 mx_microfacet_diffuse.glsl directional-albedo curve fit for Burley diffuse, provided by
    // Stephen Hill.
    float diffuse_disney_directional_albedo(const float n_dot_v)
    {
        float x = saturate(n_dot_v);
        float one_minus_x = 1.0 - x;
        float one_minus_x2 = one_minus_x * one_minus_x;
        float fit0 = 0.97619 - 0.488095 * one_minus_x2 * one_minus_x2 * one_minus_x;
        float fit1 = 1.55754 + (-2.02221 + (2.56283 - 1.06244 * x) * x) * x;
        return lerp(fit0, fit1, roughness);
    }

    [Differentiable]
    float diffuse_pdf(const float3 wo) { return max(0.0, wo.z) * float.INV_PI; }

    [Differentiable]
    float3 specular_microfacet(const float3 wi, const float3 wo, const float3 albedo, const float roughness)
    {
        float alpha = roughness * roughness;
        float3 h = normalize(wi + wo);
        float wi_dot_h = dot(wi, h);
        float D = evalNdfGGX(alpha, h.z);
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
        float3 F = eval_fresnel_schlick(albedo, float3(1.0), wi_dot_h);
        return F * (D * G * 0.25f / max(1e-6f, wi.z * wo.z));
    }

    // MaterialX 1.39 mx_microfacet_specular.glsl rational-quadratic fit to Monte Carlo data for GGX directional
    // albedo.
    float3 specular_microfacet_directional_albedo(const float n_dot_v)
    {
        float x = saturate(n_dot_v);
        float y = roughness * roughness;
        float x2 = x * x;
        float y2 = y * y;
        float4 r = float4(0.1003, 0.9345, 1.0, 1.0) + float4(-0.6303, -2.323, -1.765, 0.2281) * x
            + float4(9.748, 2.229, 8.263, 15.94) * y + float4(-2.038, -3.748, 11.53, -55.83) * x * y
            + float4(29.34, 1.424, 28.96, 13.08) * x2 + float4(-8.245, -0.7684, -7.507, 41.26) * y2
            + float4(-26.44, 1.436, -36.11, 54.9) * x2 * y + float4(19.99, 0.2913, 15.86, 300.2) * x * y2
            + float4(-5.448, 0.6286, 33.37, -285.1) * x2 * y2;
        float2 ab = saturate(r.xy / r.zw);
        return specular * ab.x + ab.y;
    }

    [Differentiable]
    bool specular_microfacet_sample(const float3 wi, const float alpha, const float2 u, out float3 wo, out float pdf)
    {
        float3 h = sampleGGX_VNDF(alpha, wi, u, pdf); // pdf = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
        float wi_dot_h = dot(wi, h);
        wo = 2.0 * wi_dot_h * h - wi;
        if (wo.z <= 0.0)
            return false;
        pdf /= (4.0 * wi_dot_h); // Jacobian of the reflection operator.
        return true;
    }

    [Differentiable]
    float specular_microfacet_pdf(const float3 wi, const float3 wo, const float alpha)
    {
        float3 h = normalize(wi + wo);
        float wi_dot_h = max(0.0, dot(wi, h));
        float pdf = evalPdfGGX_VNDF(alpha, wi, h);
        return pdf / (4.0 * wi_dot_h);
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float3 reflection = saturate(
            diffuse * diffuse_disney_directional_albedo(wi.z) + specular_microfacet_directional_albedo(wi.z)
        );
        return AlbedoContributions(reflection, 1.0 - reflection, float3(0.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi) { return RoughnessInformation(roughness, float3(0.0)); }

    public property float3 ior_as_reflectance
    {
        get {
            return specular;
        }
    }
};
