// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Unit-tint refractive GGX transmission. Fresnel belongs to material composition.
/// Uses radiance transport, eta = incident / transmitted IOR, and separable Smith masking.
public struct RefractiveSpecularMicrofacetBTDF : IBSDF, IDifferentiable
{
    public float alpha = {};
    public float eta = 1.0;

    public __init() { }
    public __init(float alpha, float eta)
    {
        this.alpha = eta == 1.0 ? 0.0 : alpha;
        this.eta = eta;
    }

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
        if (alpha == 0.0 || min(wi.z, -wo.z) < BSDF_MIN_COS_THETA)
            return BSDFEval();
        float3 h = microfacet_refraction::half_vector(wi, wo, eta);
        float vh = dot(wi, h), lh = dot(wo, h);
        if (vh <= 0.0 || lh >= 0.0 || microfacet_refraction::transmitted_cosine_squared(vh, eta) <= 0.0)
            return BSDFEval();
        float pdf = evalPdfGGX_VNDF(alpha, wi, h) * microfacet_refraction::jacobian(vh, lh, eta);
        float weight = eta * eta * evalG1GGX(alpha * alpha, -wo.z);
        return BSDFEval(float3(pdf * weight), pdf);
    }

    public bool sample(const float3 wi, const BSDFSampleInput input, const BSDFContext bc, out BSDFSample sample)
    {
        sample = {};
        if (wi.z < BSDF_MIN_COS_THETA)
            return false;
        // Equal IORs transmit straight through; avoid cancellation in the Snell/TIR calculation.
        if (eta == 1.0) {
            sample.wo = -wi;
            sample.weight = float3(1.0);
            sample.pdf = 0.0;
            sample.flags = BSDFFlags::delta_transmission;
            return true;
        }
        float half_pdf;
        float3 h = alpha == 0.0 ? float3(0.0, 0.0, 1.0) : sampleGGX_VNDF(alpha, wi, input.u_dir, half_pdf);
        float ct2 = microfacet_refraction::transmitted_cosine_squared(dot(wi, h), eta);
        if (ct2 <= 0.0)
            return false;
        sample.wo = microfacet_refraction::direction(wi, h, eta, sqrt(ct2));
        if (-sample.wo.z < BSDF_MIN_COS_THETA)
            return false;
        sample.weight = float3(eta * eta);
        if (alpha == 0.0) {
            sample.flags = BSDFFlags::delta_transmission;
            return true;
        }
        sample.pdf = eval_pdf(wi, sample.wo, bc);
        sample.weight *= evalG1GGX(alpha * alpha, -sample.wo.z);
        sample.flags = BSDFFlags::glossy_transmission;
        return sample.pdf > 0.0;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        return AlbedoContributions(float3(0.0), float3(0.0), float3(1.0), float3(0.0));
    }
    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(alpha), float3(0.0));
    }
    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }
};

/// Unit-tint thin-sheet GGX transmission with separable Smith masking.
/// Material composition supplies Fresnel and tint, independently of this sampling density.
public struct ThinSpecularMicrofacetBTDF : IBSDF, IDifferentiable
{
    public float alpha = {};

    public __init() { }
    public __init(float alpha) { this.alpha = alpha; }

    [Differentiable]
    private float distribution(float3 h)
    {
        // The cosine-only GGX expression loses alpha^2 near h.z == 1 for narrow lobes.
        // Retain the tangent components to avoid rounding a nonzero lobe into a singularity.
        float a2 = alpha * alpha;
        float d = dot(h.xy, h.xy) + a2 * h.z * h.z;
        // Divide before squaring so d*d cannot underflow for half-precision roughness.
        float ratio = alpha / d;
        return ratio * ratio * float.INV_PI;
    }

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
        if (alpha == 0.0 || min(wi.z, -wo.z) < BSDF_MIN_COS_THETA)
            return BSDFEval();
        float3 h = normalize(wi + float3(wo.x, wo.y, -wo.z));
        float D = distribution(h);
        float G1 = evalG1GGX(alpha * alpha, wi.z);
        float G2 = evalG1GGX(alpha * alpha, -wo.z);
        float pdf = D * G1 / (4.0 * wi.z);
        return BSDFEval(float3(pdf * G2), pdf);
    }

    public bool sample(const float3 wi, const BSDFSampleInput input, const BSDFContext bc, out BSDFSample sample)
    {
        sample = {};
        if (wi.z < BSDF_MIN_COS_THETA)
            return false;
        if (alpha == 0.0) {
            sample.wo = -wi;
            sample.weight = float3(1.0);
            sample.flags = BSDFFlags::delta_transmission;
            return true;
        }
        float half_pdf;
        float3 h = sampleGGX_VNDF(alpha, wi, input.u_dir, half_pdf);
        float3 reflected = 2.0 * dot(wi, h) * h - wi;
        sample.wo = float3(reflected.x, reflected.y, -reflected.z);

        BSDFEval evaluated = eval_with_pdf(wi, sample.wo, bc);
        if (evaluated.pdf <= 0.0)
            return false;
        sample.pdf = evaluated.pdf;
        sample.weight = float3(evalG1GGX(alpha * alpha, -sample.wo.z));
        sample.flags = BSDFFlags::glossy_transmission;
        return true;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        // Guide estimate; exact directional energy depends on roughness and support.
        return AlbedoContributions(float3(0.0), float3(0.0), float3(1.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(alpha), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }
};
