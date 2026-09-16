// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mtlx_snippets;

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

    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return BSDFEval();
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};
        sample.wo = -wi;
        sample.pdf = 0.0;
        sample.weight = saturate(tint);
        sample.flags = BSDFFlags::delta_transmission;
        return mx_hmax(sample.weight) > 0.0;
    }

    public override float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf(wi, wo, sg, bc).pdf;
    }

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
