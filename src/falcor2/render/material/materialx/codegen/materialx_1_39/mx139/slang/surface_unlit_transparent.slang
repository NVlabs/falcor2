// SPDX-License-Identifier: Apache-2.0

implementing mx139_snippets;

AlbedoContributions surface_unlit_transmission_albedo(float3 transmission)
{
    transmission = saturate(transmission);
    return AlbedoContributions(float3(0.0), float3(1.0) - transmission, transmission, float3(0.0));
}

public struct MxSurfaceUnlitTransparentBSDF : ILayerableBSDF
{
    public float3 tint = float3(1.0);

    public __init() { }

    public __init(float3 tint) { this.tint = max(tint, float3(0.0)); }

    public float3 throughput(const float3 wi) { return saturate(tint); }

    public float3 directional_albedo(const float3 wi) { return saturate(tint); }

    public float3 layer_opacity(const float3 wi) { return float3(0.0); }

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return float3(0.0);
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
        float3 clamped_tint = saturate(tint);
        wo = -wi;
        pdf = float.MAX;
        weight = clamped_tint;
        lobe_types = LobeTypes::delta_transmission;
        return mx_hmax(clamped_tint) > 0.0;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc) { return 0.0; }

    public AlbedoContributions eval_albedo(const float3 wi) { return surface_unlit_transmission_albedo(tint); }

    public RoughnessInformation eval_roughness(const float3 wi) { return RoughnessInformation(); }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }
};

public MxSurfaceUnlitTransparentBSDF create_mx_transparent(float3 tint)
{
    return MxSurfaceUnlitTransparentBSDF(tint);
}
