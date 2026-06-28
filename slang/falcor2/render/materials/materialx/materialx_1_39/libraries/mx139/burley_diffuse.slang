// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Burley diffuse BSDF ported from OSL/BSDL MTX.
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_burley_diffuse_impl.h

implementing mx139;

namespace mx139 {

public struct MxBurleyDiffuseBSDF : ILayerableBSDF
{
    public float3 albedo_value = {};
    public float roughness = {};

    public __init() { }

    public __init(float3 albedo_value, float roughness)
    {
        this.albedo_value = albedo_value;
        this.roughness = clamp(roughness, 0.0, 1.0);
    }

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
        return RoughnessInformation(float2(roughness, roughness), float3(0.0));
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

    // BSDL: bsdf_burley_diffuse_impl.h:27-33 fresnel
    private static float fresnel(float cos_theta, float F90)
    {
        const float x = 1.0 - cos_theta;
        return lerp(1.0, F90, pow(x, 5.0));
    }

    // BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_burley_diffuse_impl.h:36-58
    // eval_impl, translated to Falcor wi/wo naming.
    private MxBsdfSample eval_bsdf_sample(const float3 wi, const float3 wo)
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return MxBsdfSample();

        const float3 H = wi + wo;
        // BSDL uses MAX_ABS_XYZ(H) < EPSILON with EPSILON = 1e-4 (config.h:40).
        if (mx_hmax(abs(H)) < BSDL_EPSILON)
            return MxBsdfSample();

        const float3 Hn = normalize(H);
        const float cos_HI = dot(wo, Hn);
        const float cos_NO = wi.z;
        const float cos_NI = wo.z;
        const float F90 = 0.5 + 2.0 * roughness * cos_HI * cos_HI;
        const float ref_L = fresnel(cos_NI, F90);
        const float ref_V = fresnel(cos_NO, F90);
        return eval_diffuse_reflection_sample(wi, wo, albedo_value * (ref_L * ref_V), 1.0);
    }

    // BSDL: bsdf_burley_diffuse_impl.h:60-70 sample_impl
    private MxBsdfSample sample_bsdf_sample(const float3 wi, const float3 rnd)
    {
        if (wi.z <= 0.0)
            return MxBsdfSample();
        return eval_bsdf_sample(wi, square_to_cosine_hemisphere_concentric(rnd.xy));
    }
};

public MxBurleyDiffuseBSDF create_mx_burley(float3 color, float roughness, float3 normal)
{
    return MxBurleyDiffuseBSDF(color, roughness);
}

} // namespace mx139
