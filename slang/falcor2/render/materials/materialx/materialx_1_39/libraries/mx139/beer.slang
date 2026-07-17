// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mx139;

namespace mx139 {

public struct MxBeerBSDF : ILayerableBSDF
{
    public BeerBTDF inner = {};

    public __init() { }

    public __init(BeerBTDF inner) { this.inner = inner; }

    // ILayerableBSDF

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return inner.eval(wi, wo, sg, bc);
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        return inner.sample(wi, sample, sg, bc);
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc) { return inner.eval_pdf(wi, wo, bc); }

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

} // namespace mx139
