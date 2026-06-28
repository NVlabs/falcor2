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
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < MIN_COS_THETA)
            return float3(0.0);

        float3 fd = diffuse_disney(wi, wo, diffuse, roughness);
        float3 fs = specular_microfacet(wi, wo, specular, roughness);
        return (fd + fs) * wo.z;
    }

    [Differentiable]
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

        float diffuse_weight = luminance(diffuse);
        float specular_weight = max(0.001, luminance(specular));
        float norm = 1.0 / (diffuse_weight + specular_weight);
        diffuse_weight *= norm;
        specular_weight *= norm;

        float alpha = roughness * roughness;
        float u_select = no_diff next_float(sg);
        float2 u = no_diff next_float2(sg);

        if (u_select < diffuse_weight) {
            wo = square_to_cosine_hemisphere_concentric(u);
            pdf = square_to_cosine_hemisphere_concentric_pdf(wo);
            pdf = pdf * diffuse_weight + specular_weight * specular_microfacet_pdf(wi, wo, alpha);
        } else {
            if (!specular_microfacet_sample(wi, alpha, u, wo, pdf))
                return false;
            pdf = pdf * specular_weight + diffuse_weight * diffuse_pdf(wo);
        }

        if (min(wi.z, wo.z) < MIN_COS_THETA || pdf == 0.0)
            return false;

        weight = eval(wi, wo, sg, bc) / pdf;
        return true;
    }

    [Differentiable]
    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < MIN_COS_THETA)
            return 0.0;

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

        return pdf;
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
        return AlbedoContributions(float3(1.0), float3(0.0), float3(0.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi) { return RoughnessInformation(roughness, float3(0.0)); }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(1.0);
        }
    }
};
