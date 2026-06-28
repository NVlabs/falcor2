// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mx139;

import falcor2.render;
import falcor2.utils;

namespace mx139 {

public static constexpr float OSLE_DIELECTRIC_IOR_MIN = 1.001;
public static constexpr float OSLE_DIELECTRIC_IOR_MAX = 5.0;

public float clamp_dielectric_ior(float ior)
{
    // BSDL's MaterialX DielectricLobe clamps the raw material IOR before
    // converting it through globals.relative_eta(...) and before backside
    // handling. Keep this at the MX139 wrapper boundary so the Mx139 BSDF can
    // still model BSDL eta/refraction behavior internally.
    return clamp(ior, OSLE_DIELECTRIC_IOR_MIN, OSLE_DIELECTRIC_IOR_MAX);
}

public float3 mx_geomprop_tangent(SurfaceInteraction si)
{
    return si.shading_frame_ws.tangent;
}

public float3 mx_geomprop_bitangent(SurfaceInteraction si)
{
    return si.shading_frame_ws.bitangent;
}

public float3 mx_viewdirection_vector3(SurfaceInteraction si)
{
    // MaterialX <viewdirection> is defined as viewer/camera-to-surface. Falcor
    // `si.wi_ws` points surface-to-viewer, so flip it at this graph-helper
    // boundary.
    return -si.wi_ws;
}

public float mx_smoothstep_float(float low, float high, float value)
{
    // Match MaterialX GLSL/MSL and GenOSL high-first edge behavior.
    // Reversed or degenerate intervals collapse to step(high, value).
    if (value >= high)
        return 1.0;
    if (value <= low)
        return 0.0;
    return smoothstep(low, high, value);
}

public float balance_probability(float3 albedo_a, float3 albedo_b)
{
    const float weight_a = luminance(albedo_a);
    const float weight_b = luminance(albedo_b);
    float sum = weight_a + weight_b;
    return sum <= 0.0 ? 0.5 : weight_a / sum;
}

public struct MxBranchChoice
{
    public bool first = false;
    public float pdf = 0.0;

    public __init() { }
    public __init(bool first, float pdf)
    {
        this.first = first;
        this.pdf = pdf;
    }
}

public MxBranchChoice mx_choose_branch<S : ISampleGenerator>(inout S sg, float p_first)
{
    p_first = clamp(p_first, 0.0, 1.0);
    bool first = next_float(sg) < p_first;
    float choice_pdf = first ? p_first : 1.0 - p_first;
    return MxBranchChoice(first, choice_pdf);
}

AlbedoContributions from_scattering_albedo(float3 reflection, float3 transmission)
{
    float3 absorption = max(float3(0.0), float3(1.0) - reflection - transmission);
    return AlbedoContributions(reflection, absorption, transmission, float3(0.0));
}

float3 scattering_albedo_from_ibsdf(AlbedoContributions albedo)
{
    return albedo.reflection + albedo.transmission;
}

float3 layer_opacity_from_ibsdf(AlbedoContributions albedo)
{
    return saturate(float3(1.0) - albedo.transmission);
}

AlbedoContributions from_layerable_albedo(float3 sample_albedo, float3 layer_opacity)
{
    const float3 layer_transmission = saturate(float3(1.0) - layer_opacity);
    return AlbedoContributions(
        sample_albedo,
        max(float3(0.0), float3(1.0) - sample_albedo - layer_transmission),
        layer_transmission,
        float3(0.0)
    );
}

public float schlick_refraction_ior_from_color0(float3 color0)
{
    const float avg_F0 = clamp((color0.x + color0.y + color0.z) * (1.0 / 3.0), 0.0, 0.99);
    const float sqrt_F0 = sqrt(avg_F0);
    return (1.0 + sqrt_F0) / (1.0 - sqrt_F0);
}

public struct MxNullBSDF : ILayerableBSDF
{
    // ILayerableBSDF

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
        wo = float3(0.0);
        pdf = 0.0;
        weight = float3(0.0);
        lobe_types = LobeTypes::none;
        return false;
    }
    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc) { return 0.0; }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        return AlbedoContributions(float3(0.0), float3(0.0), float3(1.0), float3(0.0));
    }
    public RoughnessInformation eval_roughness(const float3 wi) { return RoughnessInformation(); }
    public float3 directional_albedo(const float3 wi) { return float3(0.0); }
    public float3 layer_opacity(const float3 wi) { return float3(0.0); }
    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }
}

static const float3 OSLE_SIMPLE_BTDF_TRANSMITTANCE = float3(1.0);

/// Untinted straight-through delta transmission used as the transparent branch
/// for Mx139 root opacity lowering. Keep this zero-state so generated opacity
/// topology can share a single pass-through BSDF shape across all layering modes.
public struct MxSimpleBTDF : ILayerableBSDF
{
    // ILayerableBSDF

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
        wo = -wi;
        pdf = float.MAX;
        weight = OSLE_SIMPLE_BTDF_TRANSMITTANCE;
        lobe_types = LobeTypes::delta_transmission;
        return true;
    }
    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc) { return 0.0; }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        return AlbedoContributions(float3(0.0), float3(0.0), OSLE_SIMPLE_BTDF_TRANSMITTANCE, float3(0.0));
    }
    public RoughnessInformation eval_roughness(const float3 wi) { return RoughnessInformation(); }
    public float3 directional_albedo(const float3 wi) { return OSLE_SIMPLE_BTDF_TRANSMITTANCE; }
    public float3 layer_opacity(const float3 wi) { return float3(0.0); }
    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }
}

/// Layerable adapter for legacy Falcor BSDFs used as Mx139 BSDFs.
public struct MxLayerableBSDF<Inner : IBSDF> : ILayerableBSDF
{
    public Inner inner = {};

    public __init() { }
    public __init(Inner inner) { this.inner = inner; }

    // ILayerableBSDF

    public float3 directional_albedo(const float3 wi) { return scattering_albedo_from_ibsdf(inner.eval_albedo(wi)); }
    public float3 layer_opacity(const float3 wi) { return layer_opacity_from_ibsdf(inner.eval_albedo(wi)); }
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return inner.eval(wi, wo, sg, bc);
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
        return inner.sample(wi, wo, pdf, weight, lobe_types, sg, bc);
    }
    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc) { return inner.eval_pdf(wi, wo, bc); }
    public AlbedoContributions eval_albedo(const float3 wi) { return inner.eval_albedo(wi); }
    public RoughnessInformation eval_roughness(const float3 wi) { return inner.eval_roughness(wi); }
    public property float3 ior_as_reflectance
    {
        get {
            return inner.ior_as_reflectance;
        }
    }
}

public struct MxWeightedBSDF<Inner : ILayerableBSDF> : ILayerableBSDF
{
    public Inner inner = {};
    public float3 weight = float3(1.0);
    // TODO(mx139-vdf): Generated MX139 VDF nodes can write this scale for
    // raster-style Beer attenuation. The current BSDF path does not transport
    // medium coefficients, so eval/sample/albedo intentionally ignore this
    // field until VDF transport is wired in.
    public float3 transmission_scale = float3(1.0);

    public __init() { }
    public __init(Inner inner, float3 weight)
    {
        this.inner = inner;
        this.weight = weight;
    }
    public __init(Inner inner, float3 weight, float3 transmission_scale)
    {
        this.inner = inner;
        this.weight = weight;
        this.transmission_scale = transmission_scale;
    }

    // ILayerableBSDF

    public float3 directional_albedo(const float3 wi)
    {
        if (scalar_weight() <= 0.0)
            return float3(0.0);
        return weight * inner.directional_albedo(wi);
    }
    public float3 layer_opacity(const float3 wi)
    {
        if (scalar_weight() <= 0.0)
            return float3(0.0);
        return saturate(weight * inner.layer_opacity(wi));
    }
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (scalar_weight() <= 0.0)
            return float3(0.0);
        return weight * inner.eval(wi, wo, sg, bc);
    }
    public bool sample<S : ISampleGenerator>(
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 sample_weight,
        out LobeTypes lobe_types,
        inout S sg,
        const BSDFContext bc
    )
    {
        wo = float3(0.0);
        pdf = 0.0;
        sample_weight = float3(0.0);
        lobe_types = LobeTypes::none;

        if (scalar_weight() <= 0.0) {
            return false;
        }
        bool valid = inner.sample(wi, wo, pdf, sample_weight, lobe_types, sg, bc);
        if (!valid) {
            wo = float3(0.0);
            pdf = 0.0;
            sample_weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        sample_weight *= weight;
        return true;
    }
    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        return scalar_weight() > 0.0 ? inner.eval_pdf(wi, wo, bc) : 0.0;
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        if (scalar_weight() <= 0.0)
            return from_layerable_albedo(float3(0.0), float3(0.0));
        return from_layerable_albedo(weight * inner.directional_albedo(wi), saturate(weight * inner.layer_opacity(wi)));
    }
    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return scalar_weight() > 0.0 ? inner.eval_roughness(wi) : RoughnessInformation();
    }
    public property float3 ior_as_reflectance
    {
        get {
            return scalar_weight() > 0.0? weight* inner.ior_as_reflectance : float3(0.0);
        }
    }

    // Implementation

    private float scalar_weight() { return clamp(mx_hmax(weight), 0.0, 1.0); }
}

public struct MxFramedWeightedBSDF<Inner : ILayerableBSDF> : ILayerableBSDF
{
    public Inner inner = {};
    public float3 weight = float3(1.0);
    // TODO(mx139-vdf): See MxWeightedBSDF. This field is
    // accepted from generated VDF layering code but ignored for OSL parity.
    public float3 transmission_scale = float3(1.0);
    public Frame frame = Frame::identity();

    public __init() { frame = Frame::identity(); }
    public __init(Inner inner, float3 weight)
    {
        this.inner = inner;
        this.weight = weight;
        this.frame = Frame::identity();
    }
    public __init(Inner inner, float3 weight, Frame frame)
    {
        this.inner = inner;
        this.weight = weight;
        this.frame = frame;
    }
    public __init(Inner inner, float3 weight, float3 transmission_scale, Frame frame)
    {
        this.inner = inner;
        this.weight = weight;
        this.transmission_scale = transmission_scale;
        this.frame = frame;
    }

    // ILayerableBSDF

    public float3 directional_albedo(const float3 wi)
    {
        if (scalar_weight() <= 0.0)
            return float3(0.0);
        return weight * inner.directional_albedo(frame.to_local(wi));
    }
    public float3 layer_opacity(const float3 wi)
    {
        if (scalar_weight() <= 0.0)
            return float3(0.0);
        return saturate(weight * inner.layer_opacity(frame.to_local(wi)));
    }
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (scalar_weight() <= 0.0)
            return float3(0.0);
        float3 local_wi = frame.to_local(wi);
        float3 local_wo = frame.to_local(wo);
        if (!validate_bump_frame_hemisphere_pair(wi, local_wi, wo, local_wo))
            return float3(0.0);
        return weight * inner.eval(local_wi, local_wo, sg, bc);
    }
    public bool sample<S : ISampleGenerator>(
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 sample_weight,
        out LobeTypes lobe_types,
        inout S sg,
        const BSDFContext bc
    )
    {
        wo = float3(0.0);
        pdf = 0.0;
        sample_weight = float3(0.0);
        lobe_types = LobeTypes::none;

        if (scalar_weight() <= 0.0) {
            return false;
        }
        const float3 local_wi = frame.to_local(wi);
        if (!validate_bump_frame_hemisphere(wi, local_wi)) {
            wo = float3(0.0);
            pdf = 0.0;
            sample_weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        float3 local_wo = float3(0.0);
        bool valid = inner.sample(local_wi, local_wo, pdf, sample_weight, lobe_types, sg, bc);
        if (!valid) {
            wo = float3(0.0);
            pdf = 0.0;
            sample_weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        wo = frame.to_global(local_wo);
        if (!validate_bump_frame_hemisphere(wo, local_wo)) {
            wo = float3(0.0);
            pdf = 0.0;
            sample_weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        sample_weight *= weight;
        return true;
    }
    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        if (scalar_weight() <= 0.0)
            return 0.0;
        const float3 local_wi = frame.to_local(wi);
        const float3 local_wo = frame.to_local(wo);
        if (!validate_bump_frame_hemisphere_pair(wi, local_wi, wo, local_wo))
            return 0.0;
        return inner.eval_pdf(local_wi, local_wo, bc);
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        if (scalar_weight() <= 0.0)
            return from_layerable_albedo(float3(0.0), float3(0.0));
        const float3 local_wi = frame.to_local(wi);
        return from_layerable_albedo(
            weight * inner.directional_albedo(local_wi),
            saturate(weight * inner.layer_opacity(local_wi))
        );
    }
    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return scalar_weight() > 0.0 ? inner.eval_roughness(frame.to_local(wi)) : RoughnessInformation();
    }
    public property float3 ior_as_reflectance
    {
        get {
            return scalar_weight() > 0.0? weight* inner.ior_as_reflectance : float3(0.0);
        }
    }

    // Implementation

    private float scalar_weight() { return clamp(mx_hmax(weight), 0.0, 1.0); }
}

public struct MxLayerBSDF<Top : ILayerableBSDF, Base : ILayerableBSDF> : ILayerableBSDF
{
    public Top top = {};
    public Base base = {};

    public __init() { }
    public __init(Top top, Base base)
    {
        this.top = top;
        this.base = base;
    }

    // ILayerableBSDF

    public float3 directional_albedo(const float3 wi)
    {
        return top.directional_albedo(wi) + base_scale(wi) * base.directional_albedo(wi);
    }
    public float3 layer_opacity(const float3 wi)
    {
        // TODO(mx139-layering): Validate this against MaterialX/OSL
        // CompositeBSDF layer opacity semantics, not a physically combined
        // coat opacity. When a
        // layer() closure is used as the top of another layer(), MX139
        // traverses through the nested layer and returns the opacity of that
        // nested layer's base branch. This preserves the bsdf_mix closure
        // weights used by the composite BSDF preparation step.
        return base.layer_opacity(wi);
    }
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return top.eval(wi, wo, sg, bc) + base_scale(wi) * base.eval(wi, wo, sg, bc);
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
        wo = float3(0.0);
        pdf = 0.0;
        weight = float3(0.0);
        lobe_types = LobeTypes::none;

        float p_top = top_sample_probability(wi);
        MxBranchChoice choice = mx_choose_branch(sg, p_top);
        if (choice.pdf <= 0.0)
            return false;
        float branch_pdf = 0.0;
        float3 branch_weight = float3(0.0);
        bool valid = choice.first ? top.sample(wi, wo, branch_pdf, branch_weight, lobe_types, sg, bc)
                                  : base.sample(wi, wo, branch_pdf, branch_weight, lobe_types, sg, bc);
        if (!valid) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        if ((uint(lobe_types) & uint(LobeTypes::delta)) != 0) {
            pdf = choice.pdf * branch_pdf;
            weight = choice.first ? branch_weight / choice.pdf : base_scale(wi) * branch_weight / choice.pdf;
            return true;
        }
        pdf = eval_pdf(wi, wo, bc);
        if (pdf <= 0.0) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        weight = eval(wi, wo, sg, bc) / pdf;
        return true;
    }
    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        float p_top = top_sample_probability(wi);
        return p_top * top.eval_pdf(wi, wo, bc) + (1.0 - p_top) * base.eval_pdf(wi, wo, bc);
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        const float3 base_weight = base_scale(wi);
        return from_layerable_albedo(
            top.directional_albedo(wi) + base_weight * base.directional_albedo(wi),
            base.layer_opacity(wi)
        );
    }
    public RoughnessInformation eval_roughness(const float3 wi)
    {
        RoughnessInformation top_r = top.eval_roughness(wi);
        return top_r;
    }
    public property float3 ior_as_reflectance
    {
        get {
            return top.ior_as_reflectance;
        }
    }

    // Implementation

    private float3 base_scale(const float3 wi) { return saturate(float3(1.0) - top.layer_opacity(wi)); }
    private float top_sample_probability(const float3 wi)
    {
        return balance_probability(top.directional_albedo(wi), base_scale(wi) * base.directional_albedo(wi));
    }
}

public struct MxMixBSDF<FG : ILayerableBSDF, BG : ILayerableBSDF> : ILayerableBSDF
{
    public FG fg = {};
    public BG bg = {};
    public float3 mix_value = float3(0.0);
    public __init() { }
    public __init(FG fg, BG bg, float mix_value)
    {
        this.fg = fg;
        this.bg = bg;
        this.mix_value = saturate(float3(mix_value));
    }
    public __init(FG fg, BG bg, float3 mix_value)
    {
        this.fg = fg;
        this.bg = bg;
        this.mix_value = saturate(mix_value);
    }

    // ILayerableBSDF

    public float3 directional_albedo(const float3 wi)
    {
        return (float3(1.0) - mix_value) * bg.directional_albedo(wi) + mix_value * fg.directional_albedo(wi);
    }
    public float3 layer_opacity(const float3 wi)
    {
        return saturate((float3(1.0) - mix_value) * bg.layer_opacity(wi) + mix_value * fg.layer_opacity(wi));
    }
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return (float3(1.0) - mix_value) * bg.eval(wi, wo, sg, bc) + mix_value * fg.eval(wi, wo, sg, bc);
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
        wo = float3(0.0);
        pdf = 0.0;
        weight = float3(0.0);
        lobe_types = LobeTypes::none;

        // OSL closure mix(a, b, amount) flattens into a*(1-amount) first,
        // followed by b*amount. Keep that CDF order so the OSL-style replay
        // remaps rx by subtracting the background interval before foreground.
        MxBranchChoice choice = mx_choose_branch(sg, bg_sample_probability(wi));
        if (choice.pdf <= 0.0)
            return false;
        float branch_pdf = 0.0;
        float3 branch_weight = float3(0.0);
        bool valid = choice.first ? bg.sample(wi, wo, branch_pdf, branch_weight, lobe_types, sg, bc)
                                  : fg.sample(wi, wo, branch_pdf, branch_weight, lobe_types, sg, bc);
        if (!valid) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        if ((uint(lobe_types) & uint(LobeTypes::delta)) != 0) {
            pdf = choice.pdf * branch_pdf;
            weight = (choice.first ? float3(1.0) - mix_value : mix_value) * branch_weight / choice.pdf;
            return true;
        }
        pdf = eval_pdf(wi, wo, bc);
        if (pdf <= 0.0) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        weight = eval(wi, wo, sg, bc) / pdf;
        return true;
    }
    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        const float p_fg = fg_sample_probability(wi);
        return p_fg * fg.eval_pdf(wi, wo, bc) + (1.0 - p_fg) * bg.eval_pdf(wi, wo, bc);
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        const float3 bg_mix = float3(1.0) - mix_value;
        return from_layerable_albedo(
            bg_mix * bg.directional_albedo(wi) + mix_value * fg.directional_albedo(wi),
            saturate(bg_mix * bg.layer_opacity(wi) + mix_value * fg.layer_opacity(wi))
        );
    }
    public RoughnessInformation eval_roughness(const float3 wi)
    {
        RoughnessInformation fr = fg.eval_roughness(wi);
        RoughnessInformation br = bg.eval_roughness(wi);
        float p_fg = fg_sample_probability(wi);
        return RoughnessInformation(lerp(br.roughness, fr.roughness, p_fg), lerp(br.scratch, fr.scratch, p_fg));
    }
    public property float3 ior_as_reflectance
    {
        get {
            return (float3(1.0) - mix_value) * bg.ior_as_reflectance + mix_value * fg.ior_as_reflectance;
        }
    }

    // Implementation

    private float fg_sample_probability(const float3 wi)
    {
        return balance_probability(
            mix_value * fg.directional_albedo(wi),
            (float3(1.0) - mix_value) * bg.directional_albedo(wi)
        );
    }
    private float bg_sample_probability(const float3 wi)
    {
        return balance_probability(
            (float3(1.0) - mix_value) * bg.directional_albedo(wi),
            mix_value * fg.directional_albedo(wi)
        );
    }
}

public struct MxAddBSDF<In1 : ILayerableBSDF, In2 : ILayerableBSDF> : ILayerableBSDF
{
    public In1 in1 = {};
    public In2 in2 = {};
    public __init() { }
    public __init(In1 in1, In2 in2)
    {
        this.in1 = in1;
        this.in2 = in2;
    }

    // ILayerableBSDF

    public float3 directional_albedo(const float3 wi)
    {
        return in1.directional_albedo(wi) + in2.directional_albedo(wi);
    }
    public float3 layer_opacity(const float3 wi) { return saturate(in1.layer_opacity(wi) + in2.layer_opacity(wi)); }
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return in1.eval(wi, wo, sg, bc) + in2.eval(wi, wo, sg, bc);
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
        wo = float3(0.0);
        pdf = 0.0;
        weight = float3(0.0);
        lobe_types = LobeTypes::none;

        float p = p_in1(wi);
        MxBranchChoice choice = mx_choose_branch(sg, p);
        if (choice.pdf <= 0.0)
            return false;
        float branch_pdf = 0.0;
        float3 branch_weight = float3(0.0);
        bool valid = choice.first ? in1.sample(wi, wo, branch_pdf, branch_weight, lobe_types, sg, bc)
                                  : in2.sample(wi, wo, branch_pdf, branch_weight, lobe_types, sg, bc);
        if (!valid) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        if ((uint(lobe_types) & uint(LobeTypes::delta)) != 0) {
            pdf = choice.pdf * branch_pdf;
            weight = branch_weight / choice.pdf;
            return true;
        }
        pdf = eval_pdf(wi, wo, bc);
        if (pdf <= 0.0) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        weight = eval(wi, wo, sg, bc) / pdf;
        return true;
    }
    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        float p = p_in1(wi);
        return p * in1.eval_pdf(wi, wo, bc) + (1.0 - p) * in2.eval_pdf(wi, wo, bc);
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        return from_layerable_albedo(
            in1.directional_albedo(wi) + in2.directional_albedo(wi),
            saturate(in1.layer_opacity(wi) + in2.layer_opacity(wi))
        );
    }
    public RoughnessInformation eval_roughness(const float3 wi)
    {
        RoughnessInformation r1 = in1.eval_roughness(wi);
        RoughnessInformation r2 = in2.eval_roughness(wi);
        float p = p_in1(wi);
        return RoughnessInformation(lerp(r2.roughness, r1.roughness, p), lerp(r2.scratch, r1.scratch, p));
    }
    public property float3 ior_as_reflectance
    {
        get {
            return saturate(in1.ior_as_reflectance + in2.ior_as_reflectance);
        }
    }

    // Implementation

    private float p_in1(const float3 wi)
    {
        return balance_probability(in1.directional_albedo(wi), in2.directional_albedo(wi));
    }
}

} // namespace mx139
