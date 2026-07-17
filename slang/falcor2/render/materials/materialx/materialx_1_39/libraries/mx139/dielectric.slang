// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// MaterialX dielectric BSDF ported from OSL/BSDL MTX.
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_dielectric_impl.h
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_dielectric_decl.h
//
// Implements the BSDLConfig::use_bvn_refraction=true path (config.h:94 default).
// "BVN" = Bounded VNDF (Eto-Tokuyoshi, "Bounded VNDF Sampling for Smith-GGX
// Reflections"), used in DielectricBSDF::eval / sample.

implementing mx139;

namespace mx139 {

public struct MxCompensatedDielectricBSDF<
    TFresnel : IMxRefractiveFresnelPolicy,
    TCompensation : IMxRefractiveCompensationPolicy,
    TScatter : IMxScatterModePolicy> : ILayerableBSDF
{
    public float2 roughness_xy;
    public TFresnel fresnel;
    public float3 reflection_tint;
    public float3 transmission_tint;
    public float3 absorption;
    public bool backfacing;

    public __init() { }

    public __init(
        float2 roughness_xy,
        TFresnel fresnel,
        float3 reflection_tint,
        float3 transmission_tint,
        float3 absorption,
        bool backfacing
    )
    {
        this.roughness_xy = roughness_xy;
        this.fresnel = fresnel;
        this.reflection_tint = reflection_tint;
        this.transmission_tint = transmission_tint;
        this.absorption = absorption;
        this.backfacing = backfacing;
    }

    // ILayerableBSDF

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        MxBsdfSample r = eval_bsdf_sample(wi, wo);
        return r.pdf > 0.0 ? r.weight * r.pdf : float3(0.0);
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};
        MxBsdfSample r = sample_bsdf_sample(wi, next_float3(sg));
        if (r.pdf <= 0.0)
            return false;
        sample.wo = r.wo;
        sample.pdf = r.pdf;
        sample.weight = r.weight;
        sample.flags = sample_lobe_types(wi, sample.wo);
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        return eval_bsdf_sample(wi, wo).pdf;
    }

    [ForceInline]
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        if (!dorefl() && !dorefr())
            return AlbedoContributions(float3(0.0), float3(1.0), float3(0.0), float3(0.0));

        const float E_ms = layer_energy(wi);
        float3 reflection;
        float3 transmission;
        if (!dorefr()) {
            reflection = reflection_tint * (1.0 - E_ms);
            transmission = float3(E_ms) * wi_absorption(wi);
        } else {
            const float3 F = fresnel.eval(abs(wi.z));
            reflection = F * (1.0 - E_ms) * reflection_tint;
            transmission = (float3(1.0) - F) * (1.0 - E_ms) * transmission_tint;
        }

        reflection = saturate(reflection);
        transmission = saturate(transmission);
        return AlbedoContributions(
            reflection,
            max(float3(0.0), float3(1.0) - reflection - transmission),
            transmission,
            float3(0.0)
        );
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(roughness_xy, float3(0.0));
    }

    [ForceInline]
    public float3 directional_albedo(const float3 wi)
    {
        return !dorefr() ? reflection_tint * (1.0 - layer_energy(wi)) : float3(1.0);
    }

    public float3 layer_opacity(const float3 wi)
    {
        const float3 layer_transmission = !dorefr() ? float3(layer_energy(wi)) * wi_absorption(wi) : float3(0.0);
        return saturate(float3(1.0) - layer_transmission);
    }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }

    // Implementation

    [ForceInline]
    private bool dorefl()
    {
        return TScatter::dorefl(reflection_tint);
    }

    [ForceInline]
    private bool dorefr() { return TScatter::dorefr(transmission_tint); }

    [ForceInline]
    private float layer_energy(float3 wi)
    {
        return TCompensation::layer_energy(
            dorefl(),
            dorefr(),
            backfacing,
            fresnel.refraction_eta(),
            microfacet_roughness(roughness_xy),
            wi.z
        );
    }

    // BSDL: bsdf_dielectric_impl.h:110-177 DielectricBSDF::eval (BVN branch)
    private MxBsdfSample spec_eval(const float3 wi, const float3 wo)
    {
        MxGGXDist dist = make_ggx_dist(roughness_xy);
        const float roughness = microfacet_roughness(roughness_xy);
        if (!dorefr()) {
            const float3 F = fresnel.eval(dot(normalize(wo + wi), wi));
            return eval_ggx_reflection_lobe(dist, F, TCompensation::reflection_scale(roughness, wi.z, F), wi, wo);
        }

        const float cos_NO = wi.z;
        const float cos_NI = wo.z;
        const bool do_reflect = dorefl();
        const bool do_refract = dorefr();
        const bool both_scatter = do_reflect && do_refract;
        if (cos_NI > 0.0) {
            if (!do_reflect)
                return MxBsdfSample();
            const float3 m = normalize(wi + wo);
            const float cos_MO = dot(m, wi);
            if (cos_MO <= 0.0)
                return MxBsdfSample();
            const float D = dist.D(m);
            const float G1 = dist.G1(wi);
            const float3 F = fresnel.eval(cos_MO);
            const float Fmax = mx_hmax(F);
            if (Fmax <= 0.0)
                return MxBsdfSample();
            const float D_refl_D = dist.D_refl_D(wi, m);
            const float D_refl = D_refl_D * D;
            // BSDL's MTX dielectric is a combined R/T helper. Advanced
            // layering can feed us as a T-only IBSDF; in that case reflection
            // is absent and the Fresnel branch probability must be one only
            // for true combined R/T BSDFs.
            const float branch_pdf = both_scatter ? Fmax : 1.0;
            const float3 out_value = F * TCompensation::refractive_reflection_scale(roughness, wi.z, F)
                * (dist.G2_G1(wo, wi) * G1 / (D_refl_D * branch_pdf));
            const float pdf = D_refl / (4.0 * cos_NO) * branch_pdf;
            return MxBsdfSample(wo, out_value, pdf, 0.0);
        } else if (cos_NI < 0.0) {
            if (!do_refract)
                return MxBsdfSample();
            float3 Ht = normalize(fresnel.refraction_eta() * wo + wi) * ((fresnel.refraction_eta() > 1.0) ? -1.0 : 1.0);
            const float cos_HO = dot(Ht, wi);
            const float cos_HI = dot(Ht, wo);
            if (cos_HO <= 0.0 || cos_HI >= 0.0)
                return MxBsdfSample();
            const float3 Ft = float3(1.0) - fresnel.eval(cos_HO);
            const float Ftmax = mx_hmax(Ft);
            if (Ht.z <= 0.0 || Ftmax <= 0.0)
                return MxBsdfSample();
            const float D = dist.D(Ht);
            const float G1 = dist.G1(wi);
            const float eta_refract = fresnel.refraction_eta();
            const float denom = cos_HI * eta_refract + cos_HO;
            const float J = (-cos_HI * cos_HO * eta_refract * eta_refract) / (wi.z * denom * denom);
            const float D_refl_D = dist.D_refl_D(wi, Ht);
            const float D_refl = D_refl_D * D;
            const float branch_pdf = both_scatter ? Ftmax : 1.0;
            const float pdf = D_refl * J * branch_pdf;
            const float3 out_value = Ft * (dist.G2_G1(float3(wo.x, wo.y, -wo.z), wi) * G1 / (D_refl_D * branch_pdf));
            return MxBsdfSample(wo, out_value, pdf, 0.0);
        }
        return MxBsdfSample();
    }

    // BSDL: bsdf_dielectric_impl.h:179-206 DielectricBSDF::sample (BVN branch)
    private MxBsdfSample spec_sample(const float3 wi, const float3 rnd)
    {
        if (!dorefr()) {
            return sample_ggx_reflection_lobe<TFresnel, TCompensation>(
                make_ggx_dist(roughness_xy),
                fresnel,
                microfacet_roughness(roughness_xy),
                wi,
                rnd
            );
        }

        MxGGXDist dist = make_ggx_dist(roughness_xy);
        float3 m = dist.sample_for_refl(wi, rnd.x, rnd.y);
        const float cos_MO = dot(wi, m);
        if (cos_MO <= 0.0)
            return MxBsdfSample();
        const float3 F = fresnel.eval(cos_MO);
        const float Fmax = mx_hmax(F);
        const bool do_reflect = dorefl();
        const bool both_scatter = do_reflect && dorefr();
        // T-only advanced BSDFs must spend all samples on refraction. The
        // Fresnel roulette is only the branch probability when reflection and
        // refraction are both present in this BSDF.
        const float reflect_probability = both_scatter ? Fmax : (do_reflect ? 1.0 : 0.0);
        const bool choose_reflect = rnd.z < reflect_probability;
        const float3 wo = choose_reflect ? reflect(-wi, m) : mx_refract(wi, m, fresnel.refraction_eta());
        if ((choose_reflect && wo.z <= 0.0) || (!choose_reflect && wo.z >= 0.0))
            return MxBsdfSample();
        return spec_eval(wi, wo);
    }

    // tint(s.wo.z) is safe when s.wo == (0,0,0) (sample failure) because the
    // companion s.pdf == 0 zeros the weight regardless of which branch is picked.
    private float3 tint(float cos_NI) { return cos_NI > 0.0 ? reflection_tint : transmission_tint; }

    // BSDL: bsdf_dielectric_impl.h:285-299 DielectricLobe::eval_impl
    private MxBsdfSample eval_bsdf_sample(const float3 wi, const float3 wo)
    {
        if (!dorefl() && !dorefr())
            return MxBsdfSample();
        MxBsdfSample s = spec_eval(wi, wo);
        if (dorefr()) {
            s.weight *= TCompensation::refraction_scale(
                dorefl(),
                dorefr(),
                backfacing,
                fresnel.refraction_eta(),
                microfacet_roughness(roughness_xy),
                wi.z
            );
        }
        s.weight *= tint(s.wo.z);
        s.roughness = microfacet_roughness(roughness_xy);
        return s;
    }

    // BSDL: bsdf_dielectric_impl.h:301-319 DielectricLobe::sample_impl
    private MxBsdfSample sample_bsdf_sample(const float3 wi, const float3 rnd)
    {
        if (!dorefl() && !dorefr())
            return MxBsdfSample();
        MxBsdfSample s = spec_sample(wi, rnd);
        if (dorefr()) {
            s.weight *= TCompensation::refraction_scale(
                dorefl(),
                dorefr(),
                backfacing,
                fresnel.refraction_eta(),
                microfacet_roughness(roughness_xy),
                wi.z
            );
        }
        if (mx_hmax(abs(s.wo)) < BSDL_EPSILON)
            return MxBsdfSample();
        s.weight *= tint(s.wo.z);
        s.roughness = microfacet_roughness(roughness_xy);
        return s;
    }

    private BSDFFlags sample_lobe_types(const float3 wi, const float3 wo)
    {
        return wi.z * wo.z > 0.0 ? BSDFFlags::glossy_reflection : BSDFFlags::glossy_transmission;
    }

    [ForceInline]
    private float3 wi_absorption(const float3 wi)
    {
        if (mx_hmax(absorption) <= 0.0 || !dorefl() || dorefr())
            return float3(1.0);
        const float sin2_NO = 1.0 - wi.z * wi.z;
        const float inveta2 = (1.0 / fresnel.refraction_eta()) * (1.0 / fresnel.refraction_eta());
        const float cos_p = sqrt(1.0 - min(1.0, inveta2 * sin2_NO));
        const float dist = 1.0 / max(cos_p, float.MIN);
        return exp(-absorption * dist);
    }
};

public typealias MxDielectricBSDF
    = MxCompensatedDielectricBSDF<MxDielectricFresnel, MxTurquinLutDielectricCompensation, MxScatterRuntimeMode>;
public typealias MxDielectricTurquinAnalyticCompensationBSDF
    = MxCompensatedDielectricBSDF<MxDielectricFresnel, MxTurquinAnalyticDielectricCompensation, MxScatterRuntimeMode>;
public typealias MxDielectricNoCompensationBSDF
    = MxCompensatedDielectricBSDF<MxDielectricFresnel, MxNoCompensationDielectricCompensation, MxScatterRuntimeMode>;

public typealias MxDielectricAiryBSDF
    = MxCompensatedDielectricBSDF<MxDielectricAiryFresnel, MxTurquinLutDielectricCompensation, MxScatterRuntimeMode>;
public typealias MxDielectricAiryTurquinAnalyticCompensationBSDF = MxCompensatedDielectricBSDF<
    MxDielectricAiryFresnel,
    MxTurquinAnalyticDielectricCompensation,
    MxScatterRuntimeMode>;
public typealias MxDielectricAiryNoCompensationBSDF = MxCompensatedDielectricBSDF<
    MxDielectricAiryFresnel,
    MxNoCompensationDielectricCompensation,
    MxScatterRuntimeMode>;

public MxDielectricBSDF create_mx_dielectric(
    float3 tint,
    float ior,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution,
    int scatter_mode,
    bool front_facing = true
)
{
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, tint);
    float material_ior = clamp_dielectric_ior(ior);
    const bool backfacing = !front_facing;
    return MxDielectricBSDF(
        roughness,
        MxDielectricFresnel(material_ior, backfacing),
        scatter_tints.reflection,
        scatter_tints.transmission,
        float3(0.0),
        backfacing
    );
}

public MxDielectricAiryBSDF create_mx_dielectric_airy(
    float3 tint,
    float ior,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution,
    int scatter_mode,
    bool front_facing = true
)
{
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, tint);
    float material_ior = clamp_dielectric_ior(ior);
    const bool backfacing = !front_facing;
    return MxDielectricAiryBSDF(
        roughness,
        MxDielectricAiryFresnel(material_ior, backfacing, thinfilm_thickness, thinfilm_ior),
        scatter_tints.reflection,
        scatter_tints.transmission,
        float3(0.0),
        backfacing
    );
}

public MxDielectricTurquinAnalyticCompensationBSDF create_mx_dielectric_comp_turquin_analytic(
    float3 tint,
    float ior,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution,
    int scatter_mode,
    bool front_facing = true
)
{
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, tint);
    float material_ior = clamp_dielectric_ior(ior);
    const bool backfacing = !front_facing;
    return MxDielectricTurquinAnalyticCompensationBSDF(
        roughness,
        MxDielectricFresnel(material_ior, backfacing),
        scatter_tints.reflection,
        scatter_tints.transmission,
        float3(0.0),
        backfacing
    );
}

public MxDielectricNoCompensationBSDF create_mx_dielectric_comp_no_compensation(
    float3 tint,
    float ior,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution,
    int scatter_mode,
    bool front_facing = true
)
{
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, tint);
    float material_ior = clamp_dielectric_ior(ior);
    const bool backfacing = !front_facing;
    return MxDielectricNoCompensationBSDF(
        roughness,
        MxDielectricFresnel(material_ior, backfacing),
        scatter_tints.reflection,
        scatter_tints.transmission,
        float3(0.0),
        backfacing
    );
}

public MxDielectricAiryTurquinAnalyticCompensationBSDF create_mx_dielectric_airy_comp_turquin_analytic(
    float3 tint,
    float ior,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution,
    int scatter_mode,
    bool front_facing = true
)
{
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, tint);
    float material_ior = clamp_dielectric_ior(ior);
    const bool backfacing = !front_facing;
    return MxDielectricAiryTurquinAnalyticCompensationBSDF(
        roughness,
        MxDielectricAiryFresnel(material_ior, backfacing, thinfilm_thickness, thinfilm_ior),
        scatter_tints.reflection,
        scatter_tints.transmission,
        float3(0.0),
        backfacing
    );
}

public MxDielectricAiryNoCompensationBSDF create_mx_dielectric_airy_comp_no_compensation(
    float3 tint,
    float ior,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution,
    int scatter_mode,
    bool front_facing = true
)
{
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, tint);
    float material_ior = clamp_dielectric_ior(ior);
    const bool backfacing = !front_facing;
    return MxDielectricAiryNoCompensationBSDF(
        roughness,
        MxDielectricAiryFresnel(material_ior, backfacing, thinfilm_thickness, thinfilm_ior),
        scatter_tints.reflection,
        scatter_tints.transmission,
        float3(0.0),
        backfacing
    );
}

} // namespace mx139
