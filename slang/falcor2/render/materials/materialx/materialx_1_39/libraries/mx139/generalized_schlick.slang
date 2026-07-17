// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// MaterialX generalized Schlick BSDF ported from OSL/BSDL MTX.
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_schlick_impl.h
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_schlick_decl.h
//
// Reuses the dielectric BVN refraction path (DielectricBSDF<SchlickFresnel> in
// BSDL); mx139 inlines the same math here with Schlick-specific Fresnel.

implementing mx139;

namespace mx139 {

public struct MxCompensatedGeneralizedSchlickBSDF<
    TFresnel : IMxRefractiveFresnelPolicy,
    TCompensation : IMxRefractiveCompensationPolicy,
    TScatter : IMxScatterModePolicy> : ILayerableBSDF
{
    public float2 roughness_xy;
    public TFresnel fresnel;
    public float3 reflection_tint;
    public float3 transmission_tint;
    public bool backfacing;

    public __init() { }

    public __init(
        float2 roughness_xy,
        TFresnel fresnel,
        float3 reflection_tint,
        float3 transmission_tint,
        bool backfacing
    )
    {
        this.roughness_xy = roughness_xy;
        this.fresnel = fresnel;
        this.reflection_tint = reflection_tint;
        this.transmission_tint = transmission_tint;
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

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        if (!dorefl() && !dorefr())
            return AlbedoContributions(float3(0.0), float3(1.0), float3(0.0), float3(0.0));

        const float E_ms = layer_energy(wi);
        float3 reflection;
        float3 transmission;
        if (!dorefr()) {
            reflection = reflection_tint * (1.0 - E_ms);
            transmission = float3(E_ms);
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

    public float3 directional_albedo(const float3 wi)
    {
        return !dorefr() ? reflection_tint * (1.0 - layer_energy(wi)) : float3(1.0);
    }

    public float3 layer_opacity(const float3 wi)
    {
        return dorefr() ? float3(1.0) : saturate(float3(1.0) - float3(layer_energy(wi)));
    }

    public property float3 ior_as_reflectance
    {
        get {
            return fresnel.eval(1.0);
        }
    }

    // Implementation

    private bool dorefl() { return TScatter::dorefl(reflection_tint); }

    private bool dorefr() { return TScatter::dorefr(transmission_tint); }

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

    private MxBsdfSample spec_eval(const float3 wi, const float3 wo)
    {
        MxGGXDist dist = make_ggx_dist(roughness_xy);
        const float roughness = microfacet_roughness(roughness_xy);
        if (!dorefr()) {
            const float3 m = normalize(wi + wo);
            const float3 F = fresnel.eval(dot(m, wi));
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
            // BSDL's MTX Schlick helper is a combined R/T helper. Advanced
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
            const float Fmax = max(0.0, 1.0 - min(Ft.x, min(Ft.y, Ft.z)));
            const float transmission_probability = both_scatter ? 1.0 : max(0.0, 1.0 - Fmax);
            // OSL's RGB Schlick sampler rejects the zero-tinted reflection branch
            // with probability Fmax. T-only Mx139 samples refraction directly.
            // Keep the virtual roulette compensation in the PDF so sample()
            // reproduces BSDL's collapsed T-only throughput even though it does
            // not spend samples on the zero-tint reflection branch.
            const float branch_pdf
                = both_scatter ? Ftmax : (transmission_probability > 0.0 ? Ftmax / transmission_probability : 0.0);
            if (branch_pdf <= 0.0)
                return MxBsdfSample();
            const float pdf = D_refl * J * branch_pdf;
            const float3 out_value = Ft * (dist.G2_G1(float3(wo.x, wo.y, -wo.z), wi) * G1 / (D_refl_D * branch_pdf));
            return MxBsdfSample(wo, out_value, pdf, 0.0);
        }
        return MxBsdfSample();
    }

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

    private float3 tint(float cos_NI) { return cos_NI > 0.0 ? reflection_tint : transmission_tint; }

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
};

public typealias MxGeneralizedSchlickBSDF
    = MxCompensatedGeneralizedSchlickBSDF<MxSchlickFresnel, MxTurquinLutDielectricCompensation, MxScatterRuntimeMode>;
public typealias MxGeneralizedSchlickTurquinAnalyticCompensationBSDF = MxCompensatedGeneralizedSchlickBSDF<
    MxSchlickFresnel,
    MxTurquinAnalyticDielectricCompensation,
    MxScatterRuntimeMode>;
public typealias MxGeneralizedSchlickNoCompensationBSDF = MxCompensatedGeneralizedSchlickBSDF<
    MxSchlickFresnel,
    MxNoCompensationDielectricCompensation,
    MxScatterRuntimeMode>;

public typealias MxGeneralizedSchlickColor82BSDF = MxCompensatedGeneralizedSchlickBSDF<
    MxSchlickColor82Fresnel,
    MxTurquinLutDielectricCompensation,
    MxScatterRuntimeMode>;
public typealias MxGeneralizedSchlickColor82TurquinAnalyticCompensationBSDF = MxCompensatedGeneralizedSchlickBSDF<
    MxSchlickColor82Fresnel,
    MxTurquinAnalyticDielectricCompensation,
    MxScatterRuntimeMode>;
public typealias MxGeneralizedSchlickColor82NoCompensationBSDF = MxCompensatedGeneralizedSchlickBSDF<
    MxSchlickColor82Fresnel,
    MxNoCompensationDielectricCompensation,
    MxScatterRuntimeMode>;

public typealias MxGeneralizedSchlickAiryBSDF = MxCompensatedGeneralizedSchlickBSDF<
    MxSchlickAiryFresnel,
    MxTurquinLutDielectricCompensation,
    MxScatterRuntimeMode>;
public typealias MxGeneralizedSchlickAiryTurquinAnalyticCompensationBSDF = MxCompensatedGeneralizedSchlickBSDF<
    MxSchlickAiryFresnel,
    MxTurquinAnalyticDielectricCompensation,
    MxScatterRuntimeMode>;
public typealias MxGeneralizedSchlickAiryNoCompensationBSDF = MxCompensatedGeneralizedSchlickBSDF<
    MxSchlickAiryFresnel,
    MxNoCompensationDielectricCompensation,
    MxScatterRuntimeMode>;

public MxSchlickFresnel make_schlick_fresnel(float3 color0, float3 color90, float exponent, bool backfacing)
{
    const float refraction_ior = schlick_refraction_ior_from_color0(color0);
    return MxSchlickFresnel(color0, color90, exponent, refraction_ior, backfacing);
}

public MxSchlickColor82Fresnel
make_schlick_color82_fresnel(float3 color0, float3 color82, float3 color90, float exponent, bool backfacing)
{
    const float refraction_ior = schlick_refraction_ior_from_color0(color0);
    return MxSchlickColor82Fresnel(color0, color82, color90, exponent, refraction_ior, backfacing);
}

public MxSchlickAiryFresnel make_schlick_airy_fresnel(
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
    bool backfacing,
    float thinfilm_thickness,
    float thinfilm_ior
)
{
    const float refraction_ior = schlick_refraction_ior_from_color0(color0);
    return MxSchlickAiryFresnel(
        color0,
        color82,
        color90,
        exponent,
        refraction_ior,
        backfacing,
        thinfilm_thickness,
        thinfilm_ior
    );
}

public MxGeneralizedSchlickBSDF create_mx_generalized_schlick(
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
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
    const bool backfacing = !front_facing;
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, float3(1.0));
    return MxGeneralizedSchlickBSDF(
        roughness,
        make_schlick_fresnel(color0, color90, exponent, backfacing),
        scatter_tints.reflection,
        scatter_tints.transmission,
        backfacing
    );
}

public MxGeneralizedSchlickColor82BSDF create_mx_generalized_schlick_color82(
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
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
    const bool backfacing = !front_facing;
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, float3(1.0));
    return MxGeneralizedSchlickColor82BSDF(
        roughness,
        make_schlick_color82_fresnel(color0, color82, color90, exponent, backfacing),
        scatter_tints.reflection,
        scatter_tints.transmission,
        backfacing
    );
}

public MxGeneralizedSchlickAiryBSDF create_mx_generalized_schlick_airy(
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
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
    const bool backfacing = !front_facing;
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, float3(1.0));
    return MxGeneralizedSchlickAiryBSDF(
        roughness,
        make_schlick_airy_fresnel(color0, color82, color90, exponent, backfacing, thinfilm_thickness, thinfilm_ior),
        scatter_tints.reflection,
        scatter_tints.transmission,
        backfacing
    );
}

public MxGeneralizedSchlickTurquinAnalyticCompensationBSDF create_mx_generalized_schlick_comp_turquin_analytic(
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
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
    const bool backfacing = !front_facing;
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, float3(1.0));
    return MxGeneralizedSchlickTurquinAnalyticCompensationBSDF(
        roughness,
        make_schlick_fresnel(color0, color90, exponent, backfacing),
        scatter_tints.reflection,
        scatter_tints.transmission,
        backfacing
    );
}

public MxGeneralizedSchlickNoCompensationBSDF create_mx_generalized_schlick_comp_no_compensation(
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
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
    const bool backfacing = !front_facing;
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, float3(1.0));
    return MxGeneralizedSchlickNoCompensationBSDF(
        roughness,
        make_schlick_fresnel(color0, color90, exponent, backfacing),
        scatter_tints.reflection,
        scatter_tints.transmission,
        backfacing
    );
}

public MxGeneralizedSchlickColor82TurquinAnalyticCompensationBSDF
create_mx_generalized_schlick_color82_comp_turquin_analytic(
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
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
    const bool backfacing = !front_facing;
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, float3(1.0));
    return MxGeneralizedSchlickColor82TurquinAnalyticCompensationBSDF(
        roughness,
        make_schlick_color82_fresnel(color0, color82, color90, exponent, backfacing),
        scatter_tints.reflection,
        scatter_tints.transmission,
        backfacing
    );
}

public MxGeneralizedSchlickColor82NoCompensationBSDF create_mx_generalized_schlick_color82_comp_no_compensation(
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
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
    const bool backfacing = !front_facing;
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, float3(1.0));
    return MxGeneralizedSchlickColor82NoCompensationBSDF(
        roughness,
        make_schlick_color82_fresnel(color0, color82, color90, exponent, backfacing),
        scatter_tints.reflection,
        scatter_tints.transmission,
        backfacing
    );
}

public MxGeneralizedSchlickAiryTurquinAnalyticCompensationBSDF create_mx_generalized_schlick_airy_comp_turquin_analytic(
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
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
    const bool backfacing = !front_facing;
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, float3(1.0));
    return MxGeneralizedSchlickAiryTurquinAnalyticCompensationBSDF(
        roughness,
        make_schlick_airy_fresnel(color0, color82, color90, exponent, backfacing, thinfilm_thickness, thinfilm_ior),
        scatter_tints.reflection,
        scatter_tints.transmission,
        backfacing
    );
}

public MxGeneralizedSchlickAiryNoCompensationBSDF create_mx_generalized_schlick_airy_comp_no_compensation(
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
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
    const bool backfacing = !front_facing;
    MxScatterTints scatter_tints = split_scatter_tints(scatter_mode, float3(1.0));
    return MxGeneralizedSchlickAiryNoCompensationBSDF(
        roughness,
        make_schlick_airy_fresnel(color0, color82, color90, exponent, backfacing, thinfilm_thickness, thinfilm_ior),
        scatter_tints.reflection,
        scatter_tints.transmission,
        backfacing
    );
}

} // namespace mx139
