// SPDX-License-Identifier: Apache-2.0

implementing rendering;

public static const float MIN_COS_THETA = 1e-6f;

/// Diffuse/specular BRDF using a simple Disney diffuse lobe plus an isotropic GGX specular lobe.
/// This is useful as target for approximate BRDF fits for sampling purposes.
public struct DiffuseSpecularBRDF : IDifferentiable
{
    float3 diffuse;  ///< Diffuse albedo.
    float3 specular; ///< Specular albedo.
    float roughness; ///< Roughness (linear).

    [Differentiable]
    public __init(float3 albedo, float3 specular, float metallic, float roughness)
    {
        this.diffuse = albedo * (1.f - metallic);
        this.specular = lerp(float3(specular * 0.08f), albedo, metallic);
        this.roughness = clamp(roughness, 0.05, 1.0); // Avoid zero roughness.
    }

    [Differentiable]
    public float3 eval(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < MIN_COS_THETA)
            return float3(0.f);

        float3 fd = diffuseDisney(wi, wo, diffuse, roughness);
        float3 fs = specularMicrofacet(wi, wo, specular, roughness);
        return (fd + fs) * wo.z;
    }

    [Differentiable]
    public bool
    sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, inout UniformSampleGenerator sg)
    {
        wo = {};
        pdf = {};
        weight = {};

        float diffuseWeight = luminance(diffuse);
        float specularWeight = max(0.001f, luminance(specular));
        float norm = 1.f / (diffuseWeight + specularWeight);
        diffuseWeight *= norm;
        specularWeight *= norm;

        float alpha = roughness * roughness;
        float uSelect = no_diff next_float(sg);
        float2 u = no_diff next_float2(sg);

        if (uSelect < diffuseWeight) {
            wo = square_to_cosine_hemisphere_concentric(u);
            pdf = square_to_cosine_hemisphere_concentric_pdf(wo);
            pdf = pdf * diffuseWeight + specularWeight * specularMicrofacetPdf(wi, wo, alpha);
        } else {
            if (!specularMicrofacetSample(wi, alpha, u, wo, pdf))
                return false;
            pdf = pdf * specularWeight + diffuseWeight * diffusePdf(wo);
        }

        if (min(wi.z, wo.z) < MIN_COS_THETA || pdf == 0.f)
            return false;

        weight = eval(wi, wo) / pdf;
        return true;
    }

    [Differentiable]
    public float eval_pdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < MIN_COS_THETA)
            return 0.f;

        float diffuseWeight = luminance(diffuse);
        float specularWeight = max(0.001f, luminance(specular));
        float norm = 1.f / (diffuseWeight + specularWeight);
        diffuseWeight *= norm;
        specularWeight *= norm;

        float alpha = roughness * roughness;
        float pdf = 0.f;

        if (diffuseWeight > 0.f)
            pdf += diffuseWeight * diffusePdf(wo);
        if (specularWeight > 0.f)
            pdf += specularWeight * specularMicrofacetPdf(wi, wo, alpha);

        return pdf;
    }

    // Internal functions

    // Based on https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
    [Differentiable]
    float3 diffuseDisney(const float3 wi, const float3 wo, const float3 albedo, const float roughness)
    {
        float3 h = normalize(wi + wo);
        float woDotH = max(0.f, dot(wo, h));
        float fd90 = 0.5f + 2.f * woDotH * woDotH * roughness;
        float fd0 = 1.f;
        float wiScatter = eval_fresnel_schlick(fd0, fd90, wi.z);
        float woScatter = eval_fresnel_schlick(fd0, fd90, wo.z);
        return albedo * (wiScatter * woScatter * float.INV_PI);
    }

    [Differentiable]
    float diffusePdf(const float3 wo) { return max(0.f, wo.z) * float.INV_PI; }

    [Differentiable]
    float3 specularMicrofacet(const float3 wi, const float3 wo, const float3 albedo, const float roughness)
    {
        float alpha = roughness * roughness;
        float3 h = normalize(wi + wo);
        float wiDotH = dot(wi, h);
        float D = evalNdfGGX(alpha, h.z);
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
        float3 F = eval_fresnel_schlick(albedo, float3(1.f), wiDotH);
        return F * (D * G * 0.25f / max(1e-6f, wi.z * wo.z));
    }

    [Differentiable]
    bool specularMicrofacetSample(const float3 wi, const float alpha, const float2 u, out float3 wo, out float pdf)
    {
        float3 h = sampleGGX_VNDF(alpha, wi, u, pdf); // pdf = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
        float wiDotH = dot(wi, h);
        wo = 2.f * wiDotH * h - wi;
        if (wo.z <= 0.f)
            return false;
        pdf /= (4.f * wiDotH); // Jacobian of the reflection operator.
        return true;
    }

    [Differentiable]
    float specularMicrofacetPdf(const float3 wi, const float3 wo, const float alpha)
    {
        float3 h = normalize(wi + wo);
        float wiDotH = max(0.f, dot(wi, h));
        float pdf = evalPdfGGX_VNDF(alpha, wi, h);
        return pdf / (4.f * wiDotH);
    }
};
