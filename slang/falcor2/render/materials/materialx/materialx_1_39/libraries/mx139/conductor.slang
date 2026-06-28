// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// MaterialX conductor BSDF ported from OSL/BSDL MTX.
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_conductor_impl.h
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_conductor_decl.h

implementing mx139;

namespace mx139 {

public struct MxCompensatedConductorBSDF<
    TFresnel : IMxReflectiveFresnelPolicy,
    TCompensation : IMxMicrofacetCompensationPolicy> : ILayerableBSDF
{
    public float2 roughness_xy;
    public TFresnel fresnel;

    public __init() { }

    public __init(float2 roughness_xy, TFresnel fresnel)
    {
        this.roughness_xy = roughness_xy;
        this.fresnel = fresnel;
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
        lobe_types = LobeTypes::glossy_reflection;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        return eval_bsdf_sample(wi, wo).pdf;
    }

    [ForceInline]
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        const float3 reflection = saturate(directional_albedo(wi));
        return AlbedoContributions(reflection, float3(1.0) - reflection, float3(0.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(roughness_xy, float3(0.0));
    }

    // Conductor's directional albedo is the BSDL average Fresnel for legacy
    // and the MaterialX thin-film directional albedo for Airy.
    [ForceInline]
    public float3 directional_albedo(const float3 wi)
    {
        return fresnel.directional_albedo(clamp(abs(wi.z), 0.0, 1.0), microfacet_roughness(roughness_xy));
    }

    public float3 layer_opacity(const float3 wi) { return float3(1.0); }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }

    // Implementation

    // BSDL: bsdf_conductor_impl.h:109-117 eval_impl, translated to Falcor
    // wi/wo naming.
    private MxBsdfSample eval_bsdf_sample(const float3 wi, const float3 wo)
    {
        const float roughness = microfacet_roughness(roughness_xy);
        const float3 F = fresnel.eval(dot(normalize(wi + wo), wi));
        MxBsdfSample s = eval_ggx_reflection_lobe(
            make_ggx_dist(roughness_xy),
            F,
            TCompensation::reflection_scale(roughness, wi.z, F),
            wi,
            wo
        );
        s.roughness = roughness;
        return s;
    }

    // BSDL: bsdf_conductor_impl.h:119-127 sample_impl
    private MxBsdfSample sample_bsdf_sample(const float3 wi, const float3 rnd)
    {
        const float roughness = microfacet_roughness(roughness_xy);
        MxBsdfSample s = sample_ggx_reflection_lobe<TFresnel, TCompensation>(
            make_ggx_dist(roughness_xy),
            fresnel,
            roughness,
            wi,
            rnd
        );
        s.roughness = roughness;
        return s;
    }
};

public typealias MxConductorBSDF = MxCompensatedConductorBSDF<MxConductorFresnel, MxTurquinLutMicrofacetCompensation>;
public typealias MxConductorTurquinAnalyticCompensationBSDF
    = MxCompensatedConductorBSDF<MxConductorFresnel, MxTurquinAnalyticMicrofacetCompensation>;
public typealias MxConductorNoCompensationBSDF
    = MxCompensatedConductorBSDF<MxConductorFresnel, MxNoCompensationMicrofacetCompensation>;

public typealias MxConductorAiryBSDF
    = MxCompensatedConductorBSDF<MxConductorAiryFresnel, MxTurquinLutMicrofacetCompensation>;
public typealias MxConductorAiryTurquinAnalyticCompensationBSDF
    = MxCompensatedConductorBSDF<MxConductorAiryFresnel, MxTurquinAnalyticMicrofacetCompensation>;
public typealias MxConductorAiryNoCompensationBSDF
    = MxCompensatedConductorBSDF<MxConductorAiryFresnel, MxNoCompensationMicrofacetCompensation>;

public typealias MxScratchConductorBSDF = MxLayerableBSDF<ScratchConductorBRDF>;

public MxScratchConductorBSDF create_mx_scratch_conductor(
    float3 ior,
    float3 extinction,
    float2 roughness,
    float3 scratchdirection,
    float depth,
    float3 mask,
    float scratch_roughness,
    float3 normal,
    float3 tangent,
    int distribution
)
{
    // Convert Beckmann roughness to the Phong coefficient expected by the
    // existing Falcor scratch-conductor BSDF, mirroring the 1.38 reference path.
    float phong_coefficient = scratch_roughness >= 0.00141421
        ? -(-2.0 + 2.0 * scratch_roughness * scratch_roughness) / (scratch_roughness * scratch_roughness)
        : 1000001.0;

    ScratchConductorBRDF inner = ScratchConductorBRDF(
        AnisotropicGGX(roughness),
        ior,
        extinction,
        normalize(float3(scratchdirection.xy, 0.0)),
        depth,
        mask,
        phong_coefficient
    );
    return MxScratchConductorBSDF(inner);
}

public MxConductorBSDF create_mx_conductor(
    float3 ior,
    float3 extinction,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution
)
{
    return MxConductorBSDF(roughness, MxConductorFresnel(ior, extinction));
}

public MxConductorAiryBSDF create_mx_conductor_airy(
    float3 ior,
    float3 extinction,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution
)
{
    return MxConductorAiryBSDF(roughness, MxConductorAiryFresnel(ior, extinction, thinfilm_thickness, thinfilm_ior));
}

public MxConductorTurquinAnalyticCompensationBSDF create_mx_conductor_comp_turquin_analytic(
    float3 ior,
    float3 extinction,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution
)
{
    return MxConductorTurquinAnalyticCompensationBSDF(roughness, MxConductorFresnel(ior, extinction));
}

public MxConductorNoCompensationBSDF create_mx_conductor_comp_no_compensation(
    float3 ior,
    float3 extinction,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution
)
{
    return MxConductorNoCompensationBSDF(roughness, MxConductorFresnel(ior, extinction));
}

public MxConductorAiryTurquinAnalyticCompensationBSDF create_mx_conductor_airy_comp_turquin_analytic(
    float3 ior,
    float3 extinction,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution
)
{
    return MxConductorAiryTurquinAnalyticCompensationBSDF(
        roughness,
        MxConductorAiryFresnel(ior, extinction, thinfilm_thickness, thinfilm_ior)
    );
}

public MxConductorAiryNoCompensationBSDF create_mx_conductor_airy_comp_no_compensation(
    float3 ior,
    float3 extinction,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution
)
{
    return MxConductorAiryNoCompensationBSDF(
        roughness,
        MxConductorAiryFresnel(ior, extinction, thinfilm_thickness, thinfilm_ior)
    );
}

} // namespace mx139
