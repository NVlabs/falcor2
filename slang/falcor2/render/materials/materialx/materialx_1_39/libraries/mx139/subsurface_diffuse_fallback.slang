// SPDX-License-Identifier: Apache-2.0
//
// MaterialX subsurface fallback: a diffuse lobe with the subsurface albedo.

implementing mx139;

namespace mx139 {

public struct MxSubsurfaceDiffuseFallbackBSDF : ILayerableBSDF
{
    public float3 albedo_value = {};

    public __init() { }

    public __init(float3 albedo_value) { this.albedo_value = albedo_value; }

    // ILayerableBSDF

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        MxBsdfSample r = eval_bsdf_sample(wi, wo);
        return r.pdf > 0.0 ? r.weight * r.pdf : float3(0.0);
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
        MxBsdfSample r = sample_bsdf_sample(wi, next_float3(sg));
        if (r.pdf <= 0.0) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        wo = r.wo;
        pdf = r.pdf;
        weight = r.weight;
        lobe_types = LobeTypes::diffuse_reflection;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        return eval_bsdf_sample(wi, wo).pdf;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        const float3 reflection = saturate(albedo_value);
        return AlbedoContributions(reflection, float3(1.0) - reflection, float3(0.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        // Subsurface fallback has no anisotropic roughness control; mark it
        // diffuse-equivalent for IBSDF.eval_roughness consumers.
        return RoughnessInformation(float2(1.0, 1.0), float3(0.0));
    }

    public float3 directional_albedo(const float3 wi) { return albedo_value; }

    public float3 layer_opacity(const float3 wi) { return float3(1.0); }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }

    // Implementation

    private MxBsdfSample eval_bsdf_sample(const float3 wi, const float3 wo)
    {
        return eval_diffuse_reflection_sample(wi, wo, albedo_value, 1.0);
    }

    private MxBsdfSample sample_bsdf_sample(const float3 wi, const float3 rnd)
    {
        if (wi.z <= 0.0)
            return MxBsdfSample();
        return eval_bsdf_sample(wi, square_to_cosine_hemisphere_concentric(rnd.xy));
    }
};

public MxSubsurfaceDiffuseFallbackBSDF create_mx_subsurface_diffuse_fallback(float3 color)
{
    return MxSubsurfaceDiffuseFallbackBSDF(color);
}

public MxSubsurfaceDiffuseFallbackBSDF
create_mx_subsurface(float3 color, float3 radius, float anisotropy, float3 normal)
{
    return create_mx_subsurface_diffuse_fallback(color);
}

} // namespace mx139
