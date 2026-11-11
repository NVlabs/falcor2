// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct GLTFSpecGlossBSDFData : IDifferentiable
{
    public float3 diffuse = {};
    public float3 specular = {};
    public float glossiness = {};
};

/// Direct glTF SpecGloss composition, using authored diffuse, F0 and glossiness.
/// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/README.md#specular---glossiness
public struct GLTFSpecGlossBSDF : IBSDF, IDifferentiable
{
    public LambertDiffuseBRDF diffuse = {};
    public SpecularMicrofacetBRDF specular = {};
    public float p_diffuse = {};
    public float p_specular = {};
    public BSDFFlags lobe_types = BSDFFlags::none;

    public __init() { }

    public __init(const float3 wi, const GLTFSpecGlossBSDFData data)
    {
        // glTF maps glossiness to perceptual roughness r = 1-g, then GGX width alpha = r^2.
        float roughness = 1.0 - saturate(data.glossiness);
        float alpha = roughness * roughness;

        // Falcor approximates sufficiently narrow GGX lobes as perfect reflection.
        // This threshold is renderer policy, not part of the glTF parameter mapping.
        bool is_delta = alpha < GLTF_MIN_GGX_ALPHA;
        if (is_delta)
            alpha = 0.0;

        // Authored specular is F0 directly. The specified scalar diffuse reduction
        // preserves diffuse hue; the specular leaf applies angle-dependent Fresnel.
        float3 f0 = saturate(data.specular);
        diffuse = LambertDiffuseBRDF(data.diffuse * (1.0 - max(f0.x, max(f0.y, f0.z))));
        specular = SpecularMicrofacetBRDF(f0, alpha, BSDFFlags::all);

        lobe_types = is_delta ? BSDFFlags::delta_reflection : BSDFFlags::glossy_reflection;
        if (any(diffuse.albedo > 0.0))
            lobe_types = BSDFFlags(uint(lobe_types) | uint(BSDFFlags::diffuse_reflection));

        // Luminance estimates choose which lobe to sample; they do not scale its BRDF.
        // View-normal Fresnel is a cheap estimate for sampling, not the evaluation angle.
        p_diffuse = luminance(diffuse.albedo);
        p_specular = luminance(eval_fresnel_schlick(f0, float3(1.0), wi.z));

        // Even black F0 can reflect oblique microfacets. Keep that lobe sampleable
        // when the view-normal estimate is zero; 0.01 is a proposal weight, not reflectance.
        if (p_specular == 0.0)
            p_specular = 0.01;

        // Normalize the two proposal weights into selection probabilities.
        float norm = p_diffuse + p_specular;
        if (norm > 0.0) {
            norm = 1.0 / norm;
            p_diffuse *= norm;
            p_specular *= norm;
        }
    }

    public BSDFFlags get_lobe_types() { return lobe_types; }

    [Differentiable]
    public override float3 eval(
        const float3 wi,
        const float3 wo,
        const BSDFContext bc,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        return eval_with_pdf(wi, wo, bc, component).value;
    }

    [Differentiable]
    public BSDFEval eval_with_pdf(
        const float3 wi,
        const float3 wo,
        const BSDFContext bc,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        BSDFEval result = {};

        // Add physical lobe values, but mix their PDFs by selection probability.
        // The leaf values already include the outgoing cosine.
        if (p_diffuse > 0.0 && (component == BSDFComponent::full_response || uint(component) == 0)) {
            const BSDFEval lobe = diffuse.eval_with_pdf(wi, wo, bc);
            result.value += lobe.value;
            result.pdf += p_diffuse * lobe.pdf;
        }
        if (p_specular > 0.0 && (component == BSDFComponent::full_response || uint(component) == 1)) {
            const BSDFEval lobe = specular.eval_with_pdf(wi, wo, bc);
            result.value += lobe.value;
            result.pdf += p_specular * lobe.pdf;
        }

        return result;
    }

    public bool sample(const float3 wi, const BSDFSampleInput input, const BSDFContext bc, out BSDFSample sample)
    {
        sample = {};

        // As in Standard, weight estimates the chosen physical component divided
        // by its selection probability. Non-delta pdf reports the marginal mixture.
        // It is intentionally not a full-mixture eval/pdf weight.
        // Neither child consumes u_lobe, so the input does not need remapping.
        if (input.u_lobe < p_diffuse) {
            if (!diffuse.sample(wi, input, bc, sample))
                return false;
            sample.weight *= 1.0 / p_diffuse;
            sample.component = BSDFComponent(0);
            sample.pdf *= p_diffuse;
            if (p_specular > 0.0)
                sample.pdf += p_specular * specular.eval_pdf(wi, sample.wo, bc);
        } else if (input.u_lobe < p_diffuse + p_specular) {
            if (!specular.sample(wi, input, bc, sample))
                return false;
            sample.weight *= 1.0 / p_specular;
            sample.component = BSDFComponent(1);
            sample.pdf *= p_specular;

            // Delta samples carry integrated event weights and a zero-PDF sentinel;
            // do not add a continuous diffuse density to that discrete event.
            if (sample.has_flag(BSDFFlags::delta))
                return true;
            if (p_diffuse > 0.0)
                sample.pdf += p_diffuse * diffuse.eval_pdf(wi, sample.wo, bc);
        } else {
            return false;
        }
        return true;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        // A bounded guide estimate: the specular leaf uses view-normal Fresnel,
        // not a rough-GGX hemispherical integral. There is no transmitted lobe.
        float3 reflection = saturate(diffuse.albedo + specular.eval_albedo(wi).reflection);
        return AlbedoContributions(reflection, 1.0 - reflection, float3(0.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        // This BSDF query reports GGX width, unlike the material's perceptual roughness guide.
        return RoughnessInformation(float2(specular.alpha), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            return specular.albedo;
        }
    }
};
