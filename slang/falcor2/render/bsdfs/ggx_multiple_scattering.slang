// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct GLTFScatteringGlobals
{
    public DescriptorHandle<Texture2D> energy;
    public DescriptorHandle<Texture2D> sheen_energy;
    public DescriptorHandle<SamplerState> sampler;
};

public GLTFScatteringGlobals gltf_scattering_globals;

/// Directional and cosine-average unit-Fresnel energy for correlated-Smith GGX.
public float2 gltf_ggx_energy(float cosine, float roughness)
{
    const float size = 64.0;
    float2 grid = float2(sqrt(saturate(cosine)), saturate(roughness));
    float2 uv = (grid * (size - 1.0) + 0.5) / size;
    return gltf_scattering_globals.energy.SampleLevel(gltf_scattering_globals.sampler, uv, 0.0).rg;
}

/// Reciprocal Kulla-Conty energy compensation for a reflective conductor.
/// The angular shape is an approximation; it does not simulate microfacet random walks.
public struct GGXMultipleScatteringBRDF : IBSDF, IDifferentiable
{
    public float roughness = {};
    public float3 coefficient = {};
    public float missing_incident_energy = {};
    public float average_energy = 1.0;

    public __init() { }
    public __init(float3 wi, float roughness, float3 f0)
    {
        this.roughness = roughness;
        float2 energy = gltf_ggx_energy(wi.z, roughness);
        missing_incident_energy = 1.0 - energy.x;
        average_energy = energy.y;
        float3 average_fresnel = (1.0 + 20.0 * saturate(f0)) / 21.0;
        coefficient
            = average_fresnel * average_fresnel * average_energy / (1.0 - average_fresnel * (1.0 - average_energy));
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
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA || average_energy >= 1.0)
            return BSDFEval();
        float2 outgoing = no_diff gltf_ggx_energy(wo.z, roughness);
        float outgoing_energy = outgoing.x;
        float pdf = wo.z * float.INV_PI;
        float3 value = coefficient * missing_incident_energy * (1.0 - outgoing_energy) / (1.0 - average_energy) * pdf;
        return BSDFEval(value, pdf);
    }

    public bool sample(const float3 wi, const BSDFSampleInput input, const BSDFContext bc, out BSDFSample sample)
    {
        sample = {};
        sample.wo = square_to_cosine_hemisphere_concentric(input.u_dir);
        sample.pdf = square_to_cosine_hemisphere_concentric_pdf(sample.wo);
        sample.flags = BSDFFlags::diffuse_reflection;
        if (min(wi.z, sample.wo.z) < BSDF_MIN_COS_THETA)
            return false;
        sample.weight = float3(1.0);

        BSDFEval evaluated = eval_with_pdf(wi, sample.wo, bc);
        if (evaluated.pdf <= 0.0)
            return false;
        sample.weight = evaluated.value / evaluated.pdf;
        sample.flags = BSDFFlags::glossy_reflection;
        return true;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float3 reflected = coefficient * missing_incident_energy;
        return AlbedoContributions(reflected, 1.0 - reflected, float3(0.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(roughness * roughness), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }
};
