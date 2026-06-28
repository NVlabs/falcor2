// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Roughness^2 below this threshold is treated as a delta microfacet lobe.
public static const float STANDARD_MIN_GGX_ALPHA = 0.0064;

public struct StandardBSDFData : IDifferentiable
{
    /// Diffuse reflection albedo.
    public float3 diffuse = {};
    /// Specular reflectance at normal incidence.
    public float3 specular = {};
    /// Perceptual roughness before remapping to GGX alpha.
    public float roughness = {};
    /// Metallic blend parameter.
    public float metallic = {};
    /// Relative index of refraction (eta_i / eta_t).
    public float eta = {};
    /// Transmission color.
    public float3 transmission = float3(1.0);
    /// Blend between diffuse reflection and diffuse transmission.
    public float diffuse_transmission = {};
    /// Blend between opaque dielectric BRDF and specular transmissive BSDF.
    public float specular_transmission = {};
    /// True for thin-walled transmission.
    public bool thin_walled = {};
};

/// Falcor-style prepared standard BSDF.
public struct StandardBSDF : IBSDF, IDifferentiable
{
    DisneyDiffuseBRDF diffuse_reflection = {};
    LambertDiffuseBTDF diffuse_transmission = {};
    SpecularMicrofacetBRDF specular_reflection = {};
    SpecularMicrofacetBSDF specular_transmission = {};

    float diff_trans = {};
    float spec_trans = {};
    float roughness = {};
    float metallic = {};
    LobeTypes lobe_types = LobeTypes::none;

    float p_diffuse_reflection = {};
    float p_diffuse_transmission = {};
    float p_specular_reflection = {};
    float p_specular_transmission = {};

    public __init() { }

    public __init(const float3 wi, const StandardBSDFData data)
    {
        // Clamp artist/material controls before deriving lobe weights.
        roughness = saturate(data.roughness);
        metallic = saturate(data.metallic);
        diff_trans = saturate(data.diffuse_transmission);
        spec_trans = saturate(data.specular_transmission);

        // Thick transmission stores Beer attenuation, so convert it to an interface albedo.
        float3 transmission_albedo = data.thin_walled ? saturate(data.transmission) : sqrt(saturate(data.transmission));

        // The microfacet helpers treat alpha == 0 as a delta lobe.
        float alpha = roughness * roughness;
        bool is_delta = alpha < STANDARD_MIN_GGX_ALPHA;
        if (is_delta)
            alpha = 0.0;

        // Build the four physical lobes represented by the standard material.
        diffuse_reflection = DisneyDiffuseBRDF(data.diffuse, roughness);
        diffuse_transmission = LambertDiffuseBTDF(transmission_albedo);
        specular_reflection = SpecularMicrofacetBRDF(data.specular, alpha, LobeTypes::all);
        specular_transmission
            = SpecularMicrofacetBSDF(transmission_albedo, data.eta == 1.0 ? 0.0 : alpha, data.eta, LobeTypes::all);

        lobe_types = is_delta ? LobeTypes::delta_reflection : LobeTypes::glossy_reflection;
        if (any(data.diffuse > 0.0) && spec_trans < 1.0) {
            if (diff_trans < 1.0)
                lobe_types = LobeTypes(uint(lobe_types) | uint(LobeTypes::diffuse_reflection));
            if (diff_trans > 0.0)
                lobe_types = LobeTypes(uint(lobe_types) | uint(LobeTypes::diffuse_transmission));
        }
        if (spec_trans > 0.0)
            lobe_types = LobeTypes(
                uint(lobe_types) | uint(is_delta ? LobeTypes::delta_transmission : LobeTypes::glossy_transmission)
            );

        // Split selection probability between diffuse dielectric, metallic/specular reflection,
        // and transmissive specular lobes. Fresnel at wi biases reflection sampling.
        float metallic_brdf = metallic * (1.0 - spec_trans);
        float dielectric_bsdf = (1.0 - metallic) * (1.0 - spec_trans);
        float specular_bsdf = spec_trans;

        float diffuse_weight = luminance(data.diffuse);
        float specular_weight = luminance(eval_fresnel_schlick(data.specular, float3(1.0), wi.z));

        p_diffuse_reflection = diffuse_weight * dielectric_bsdf * (1.0 - diff_trans);
        p_diffuse_transmission = diffuse_weight * dielectric_bsdf * diff_trans;
        p_specular_reflection = specular_weight * (metallic_brdf + dielectric_bsdf);
        p_specular_transmission = specular_bsdf;

        // Normalize lobe selection probabilities; zero leaves an invalid, non-sampling BSDF.
        float norm = p_diffuse_reflection + p_diffuse_transmission + p_specular_reflection + p_specular_transmission;
        if (norm > 0.0) {
            norm = 1.0 / norm;
            p_diffuse_reflection *= norm;
            p_diffuse_transmission *= norm;
            p_specular_reflection *= norm;
            p_specular_transmission *= norm;
        }
    }

    public LobeTypes get_lobe_types() { return lobe_types; }

    [Differentiable]
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        float3 result = float3(0.0);
        if (p_diffuse_reflection > 0.0)
            result += (1.0 - spec_trans) * (1.0 - diff_trans) * diffuse_reflection.eval(wi, wo, sg, bc);
        if (p_diffuse_transmission > 0.0)
            result += (1.0 - spec_trans) * diff_trans * diffuse_transmission.eval(wi, wo, sg, bc);
        if (p_specular_reflection > 0.0)
            result += (1.0 - spec_trans) * specular_reflection.eval(wi, wo, sg, bc);
        if (p_specular_transmission > 0.0)
            result += spec_trans * specular_transmission.eval(wi, wo, sg, bc);
        return result;
    }

    public bool sample<S : ISampleGenerator>(
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 weight,
        out LobeTypes sampled_lobe_types,
        inout S sg,
        const BSDFContext bc
    )
    {
        wo = {};
        pdf = {};
        weight = {};
        sampled_lobe_types = LobeTypes::none;

        float u_select = next_float(sg);

        // Sample one lobe using the prepared probabilities, then fold in the other
        // compatible lobe PDFs so the returned PDF matches eval_pdf().
        if (u_select < p_diffuse_reflection) {
            if (!diffuse_reflection.sample(wi, wo, pdf, weight, sampled_lobe_types, sg, bc))
                return false;

            weight *= (1.0 - spec_trans) * (1.0 - diff_trans) / p_diffuse_reflection;
            pdf *= p_diffuse_reflection;

            if (p_specular_reflection > 0.0)
                pdf += p_specular_reflection * specular_reflection.eval_pdf(wi, wo, bc);
            if (p_specular_transmission > 0.0)
                pdf += p_specular_transmission * specular_transmission.eval_pdf(wi, wo, bc);
        } else if (u_select < p_diffuse_reflection + p_diffuse_transmission) {
            if (!diffuse_transmission.sample(wi, wo, pdf, weight, sampled_lobe_types, sg, bc))
                return false;

            weight *= (1.0 - spec_trans) * diff_trans / p_diffuse_transmission;
            pdf *= p_diffuse_transmission;

            if (p_specular_transmission > 0.0)
                pdf += p_specular_transmission * specular_transmission.eval_pdf(wi, wo, bc);
        } else if (u_select < p_diffuse_reflection + p_diffuse_transmission + p_specular_reflection) {
            if (!specular_reflection.sample(wi, wo, pdf, weight, sampled_lobe_types, sg, bc))
                return false;

            weight *= (1.0 - spec_trans) / p_specular_reflection;
            pdf *= p_specular_reflection;

            if (p_diffuse_reflection > 0.0)
                pdf += p_diffuse_reflection * diffuse_reflection.eval_pdf(wi, wo, bc);
            if (p_specular_transmission > 0.0)
                pdf += p_specular_transmission * specular_transmission.eval_pdf(wi, wo, bc);
        } else if (p_specular_transmission > 0.0) {
            if (!specular_transmission.sample(wi, wo, pdf, weight, sampled_lobe_types, sg, bc))
                return false;

            weight *= spec_trans / p_specular_transmission;
            pdf *= p_specular_transmission;

            if (p_diffuse_reflection > 0.0)
                pdf += p_diffuse_reflection * diffuse_reflection.eval_pdf(wi, wo, bc);
            if (p_diffuse_transmission > 0.0)
                pdf += p_diffuse_transmission * diffuse_transmission.eval_pdf(wi, wo, bc);
            if (p_specular_reflection > 0.0)
                pdf += p_specular_reflection * specular_reflection.eval_pdf(wi, wo, bc);
        } else {
            return false;
        }

        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        float pdf = 0.0;
        if (p_diffuse_reflection > 0.0)
            pdf += p_diffuse_reflection * diffuse_reflection.eval_pdf(wi, wo, bc);
        if (p_diffuse_transmission > 0.0)
            pdf += p_diffuse_transmission * diffuse_transmission.eval_pdf(wi, wo, bc);
        if (p_specular_reflection > 0.0)
            pdf += p_specular_reflection * specular_reflection.eval_pdf(wi, wo, bc);
        if (p_specular_transmission > 0.0)
            pdf += p_specular_transmission * specular_transmission.eval_pdf(wi, wo, bc);
        return pdf;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        AlbedoContributions result = AlbedoContributions(float3(0.0), float3(0.0), float3(0.0), float3(0.0));
        if (p_diffuse_reflection > 0.0)
            result += (1.0 - spec_trans) * (1.0 - diff_trans) * diffuse_reflection.eval_albedo(wi);
        if (p_diffuse_transmission > 0.0)
            result += (1.0 - spec_trans) * diff_trans * diffuse_transmission.eval_albedo(wi);
        if (p_specular_reflection > 0.0)
            result += (1.0 - spec_trans) * specular_reflection.eval_albedo(wi);
        if (p_specular_transmission > 0.0)
            result += spec_trans * specular_transmission.eval_albedo(wi);
        return result;
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        float alpha = specular_reflection.alpha;
        return RoughnessInformation(float2(alpha < STANDARD_MIN_GGX_ALPHA ? 0.0 : alpha), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            return specular_reflection.albedo;
        }
    }
};

/// Test adapter that owns raw data and constructs the prepared StandardBSDF internally.
public struct TestStandardBSDF : IBSDF
{
    public StandardBSDFData data = {};

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        StandardBSDF bsdf = StandardBSDF(wi, data);
        return bsdf.eval(wi, wo, sg, bc);
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
        StandardBSDF bsdf = StandardBSDF(wi, data);
        return bsdf.sample(wi, wo, pdf, weight, lobe_types, sg, bc);
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        StandardBSDF bsdf = StandardBSDF(wi, data);
        return bsdf.eval_pdf(wi, wo, bc);
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        StandardBSDF bsdf = StandardBSDF(wi, data);
        return bsdf.eval_albedo(wi);
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        StandardBSDF bsdf = StandardBSDF(wi, data);
        return bsdf.eval_roughness(wi);
    }

    public property float3 ior_as_reflectance
    {
        get {
            return data.specular;
        }
    }
};
