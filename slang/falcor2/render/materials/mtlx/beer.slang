// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mtlx;

namespace mtlx {

public struct MxBeerBSDF : ILayerableBSDF
{
    public BeerBTDF inner = {};

    public __init() { }

    public __init(BeerBTDF inner) { this.inner = inner; }

    // ILayerableBSDF


    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override float3 eval(
        const float3 wi,
        const float3 wo,
        const BSDFContext bc,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        return eval_with_pdf(wi, wo, bc, component).value;
    }

    public override float eval_pdf(
        const float3 wi,
        const float3 wo,
        const BSDFContext bc,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        return eval_with_pdf(wi, wo, bc, component).pdf;
    }

    public BSDFEval eval_with_pdf(
        const float3 wi,
        const float3 wo,
        const BSDFContext bc,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        return inner.eval_with_pdf(wi, wo, bc);
    }

    public bool sample(const float3 wi, const BSDFSampleInput input, const BSDFContext bc, out BSDFSample sample)
    {
        return inner.sample(wi, input, bc, sample);
    }


    public AlbedoContributions eval_albedo(const float3 wi) { return inner.eval_albedo(wi); }

    public RoughnessInformation eval_roughness(const float3 wi) { return inner.eval_roughness(wi); }

    public float3 directional_albedo(const float3 wi) { return inner.eval_albedo(wi).transmission; }

    public float3 layer_opacity(const float3 wi)
    {
        // Beer is not a BSDL through-material transmissive lobe for transmits[]
        // purposes, but its absorption must still reduce layer pass-through.
        // Reporting full pass-through here makes layer(layer(glaze, beer), base)
        // look transparent to the outer layer and leaks the base diffuse.
        return saturate(float3(1.0) - inner.eval_albedo(wi).transmission);
    }

    public property float3 ior_as_reflectance
    {
        get {
            return inner.ior_as_reflectance;
        }
    }
}

public MxBeerBSDF create_mx_beer(float3 absorption, float pathshortening)
{
    return MxBeerBSDF(BeerBTDF(absorption, pathshortening));
}

} // namespace mtlx
