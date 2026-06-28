// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Translucent BSDF ported from OSL/BSDL MTX.
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_translucent_impl.h

implementing mx139;

namespace mx139 {

public struct MxTranslucentBSDF : ILayerableBSDF
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
        lobe_types = LobeTypes::diffuse_transmission;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        return eval_bsdf_sample(wi, wo).pdf;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        const float3 transmission = saturate(albedo_value);
        return AlbedoContributions(float3(0.0), float3(0.0), transmission, float3(1.0) - transmission);
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        // Translucent has no roughness control; treat as fully diffuse.
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

    // BSDL: bsdf_translucent_impl.h:25-34 eval_impl
    // BSDL ignores the view direction; we keep that contract after translating
    // to Falcor wi/wo naming.
    private MxBsdfSample eval_bsdf_sample(const float3 wi, const float3 wo)
    {
        if (wo.z >= 0.0)
            return MxBsdfSample();
        return MxBsdfSample(wo, albedo_value, abs(wo.z) * float.INV_PI, 1.0);
    }

    // BSDL: bsdf_translucent_impl.h:36-44 sample_impl
    private MxBsdfSample sample_bsdf_sample(const float3 wi, const float3 rnd)
    {
        float3 wo = square_to_cosine_hemisphere_concentric(rnd.xy);
        wo.z = -wo.z;
        return eval_bsdf_sample(wi, wo);
    }
};

public MxTranslucentBSDF create_mx_translucent(float3 color, float3 normal)
{
    return MxTranslucentBSDF(color);
}

} // namespace mx139
