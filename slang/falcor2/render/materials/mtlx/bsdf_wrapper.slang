// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mtlx;

import falcor2.render;
import falcor2.utils;

namespace mtlx {

public static constexpr float OSLE_DIELECTRIC_IOR_MIN = 1.001;
public static constexpr float OSLE_DIELECTRIC_IOR_MAX = 5.0;

public float clamp_dielectric_ior(float ior)
{
    // BSDL's MaterialX DielectricLobe clamps the raw material IOR before
    // converting it through globals.relative_eta(...) and before backside
    // handling. Keep this at the MTLX wrapper boundary so the Mtlx BSDF can
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

/// Reconstruct a partition at boundaries that provide only scattering albedo.
/// This fallback assigns all unattributed energy to upper-side absorption. Callers
/// with full AlbedoContributions should combine them directly when the operation
/// defines an exact combination.
public AlbedoContributions from_scattering_albedo(float3 reflection, float3 transmission)
{
    float3 absorption = max(float3(0.0), float3(1.0) - reflection - transmission);
    return AlbedoContributions(reflection, absorption, transmission, float3(0.0));
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

    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    public override float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf(wi, wo, sg, bc).pdf;
    }

    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return BSDFEval();
    }
    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};
        return false;
    }
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
/// for Mtlx root opacity lowering. Keep this zero-state so generated opacity
/// topology can share a single pass-through BSDF shape across all layering modes.
public struct MxSimpleBTDF : ILayerableBSDF
{
    // ILayerableBSDF

    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    public override float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf(wi, wo, sg, bc).pdf;
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
        sample.weight = OSLE_SIMPLE_BTDF_TRANSMITTANCE;
        sample.flags = BSDFFlags::delta_transmission;
        return true;
    }
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

/// Layerable adapter for legacy Falcor BSDFs used as Mtlx BSDFs.
public struct MxLayerableBSDF<Inner : IBSDF> : ILayerableBSDF
{
    public Inner inner = {};

    public __init() { }
    public __init(Inner inner) { this.inner = inner; }

    // ILayerableBSDF

    public float3 directional_albedo(const float3 wi)
    {
        AlbedoContributions albedo = inner.eval_albedo(wi);
        return albedo.reflection + albedo.transmission;
    }
    public float3 layer_opacity(const float3 wi) { return saturate(float3(1.0) - inner.eval_albedo(wi).transmission); }
    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    public override float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf(wi, wo, sg, bc).pdf;
    }

    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return inner.eval_with_pdf(wi, wo, sg, bc);
    }
    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        return inner.sample(wi, sample, sg, bc);
    }
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
    // TODO(mtlx-vdf): Generated MTLX VDF nodes can write this scale for
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
    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    public override float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf(wi, wo, sg, bc).pdf;
    }

    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (scalar_weight() <= 0.0)
            return BSDFEval();
        BSDFEval result = inner.eval_with_pdf(wi, wo, sg, bc);
        result.value *= weight;
        return result;
    }
    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        if (scalar_weight() <= 0.0)
            return false;
        bool valid = inner.sample(wi, sample, sg, bc);
        if (!valid)
            return false;
        sample.weight *= weight;
        return true;
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        if (scalar_weight() <= 0.0)
            return from_scattering_albedo(float3(0.0), float3(0.0));
        AlbedoContributions albedo = inner.eval_albedo(wi);
        return from_scattering_albedo(weight * albedo.reflection, weight * albedo.transmission);
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
    // TODO(mtlx-vdf): See MxWeightedBSDF. This field is
    // accepted from generated VDF layering code but ignored for OSL parity.
    public float3 transmission_scale = float3(1.0);
    public float material_transmission = 0.0;
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
    public __init(Inner inner, float3 weight, float3 transmission_scale, float material_transmission, Frame frame)
    {
        this.inner = inner;
        this.weight = weight;
        this.transmission_scale = transmission_scale;
        this.material_transmission = material_transmission;
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
    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    public override float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf(wi, wo, sg, bc).pdf;
    }

    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (scalar_weight() <= 0.0)
            return BSDFEval();
        float3 local_wi = frame.to_local(wi);
        float3 local_wo = frame.to_local(wo);
        BSDFEval result = inner.eval_with_pdf(local_wi, local_wo, sg, bc);
        result.value *= weight;
        return result;
    }
    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        if (scalar_weight() <= 0.0)
            return false;
        const float3 local_wi = frame.to_local(wi);
        BSDFSample local_sample;
        bool valid = inner.sample(local_wi, local_sample, sg, bc);
        if (!valid)
            return false;
        sample = local_sample;
        sample.wo = frame.to_global(local_sample.wo);
        sample.weight *= weight;
        return true;
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        if (scalar_weight() <= 0.0)
            return from_scattering_albedo(float3(0.0), float3(0.0));
        const float3 local_wi = frame.to_local(wi);
        AlbedoContributions albedo = inner.eval_albedo(local_wi);
        return from_scattering_albedo(weight * albedo.reflection, weight * albedo.transmission);
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
        // TODO(mtlx-layering): Validate this against MaterialX/OSL
        // CompositeBSDF layer opacity semantics, not a physically combined
        // coat opacity. When a
        // layer() closure is used as the top of another layer(), MTLX
        // traverses through the nested layer and returns the opacity of that
        // nested layer's base branch. This preserves the bsdf_mix closure
        // weights used by the composite BSDF preparation step.
        return base.layer_opacity(wi);
    }
    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    public override float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf(wi, wo, sg, bc).pdf;
    }

    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        BSDFEval top_result = top.eval_with_pdf(wi, wo, sg, bc);
        BSDFEval base_result = base.eval_with_pdf(wi, wo, sg, bc);
        float p_top = top_sample_probability(wi);
        return BSDFEval(
            top_result.value + base_scale(wi) * base_result.value,
            p_top * top_result.pdf + (1.0 - p_top) * base_result.pdf
        );
    }
    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        float p_top = top_sample_probability(wi);
        MxBranchChoice choice = mx_choose_branch(sg, p_top);
        if (choice.pdf <= 0.0)
            return false;
        bool valid = choice.first ? top.sample(wi, sample, sg, bc) : base.sample(wi, sample, sg, bc);
        if (!valid)
            return false;
        if (sample.has_flag(BSDFFlags::delta)) {
            sample.pdf = choice.pdf * sample.pdf;
            sample.weight = choice.first ? sample.weight / choice.pdf : base_scale(wi) * sample.weight / choice.pdf;
            return true;
        }
        const BSDFEval eval_result = eval_with_pdf(wi, sample.wo, sg, bc);
        sample.pdf = eval_result.pdf;
        if (sample.pdf <= 0.0)
            return false;
        sample.weight = eval_result.value / sample.pdf;
        return true;
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        const float3 base_weight = base_scale(wi);
        AlbedoContributions top_albedo = top.eval_albedo(wi);
        AlbedoContributions base_albedo = base.eval_albedo(wi);
        return from_scattering_albedo(
            top_albedo.reflection + base_weight * base_albedo.reflection,
            top_albedo.transmission + base_weight * base_albedo.transmission
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
    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    public override float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf(wi, wo, sg, bc).pdf;
    }

    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        BSDFEval bg_result = bg.eval_with_pdf(wi, wo, sg, bc);
        BSDFEval fg_result = fg.eval_with_pdf(wi, wo, sg, bc);
        const float p_fg = fg_sample_probability(wi);
        return BSDFEval(
            (float3(1.0) - mix_value) * bg_result.value + mix_value * fg_result.value,
            p_fg * fg_result.pdf + (1.0 - p_fg) * bg_result.pdf
        );
    }
    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        // OSL closure mix(a, b, amount) flattens into a*(1-amount) first,
        // followed by b*amount. Keep that CDF order so the OSL-style replay
        // remaps rx by subtracting the background interval before foreground.
        MxBranchChoice choice = mx_choose_branch(sg, bg_sample_probability(wi));
        if (choice.pdf <= 0.0)
            return false;
        bool valid = choice.first ? bg.sample(wi, sample, sg, bc) : fg.sample(wi, sample, sg, bc);
        if (!valid)
            return false;
        if (sample.has_flag(BSDFFlags::delta)) {
            sample.pdf = choice.pdf * sample.pdf;
            sample.weight = (choice.first ? float3(1.0) - mix_value : mix_value) * sample.weight / choice.pdf;
            return true;
        }
        const BSDFEval eval_result = eval_with_pdf(wi, sample.wo, sg, bc);
        sample.pdf = eval_result.pdf;
        if (sample.pdf <= 0.0)
            return false;
        sample.weight = eval_result.value / sample.pdf;
        return true;
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        const float3 bg_mix = float3(1.0) - mix_value;
        AlbedoContributions bg_albedo = bg.eval_albedo(wi);
        AlbedoContributions fg_albedo = fg.eval_albedo(wi);
        return AlbedoContributions(
            bg_mix * bg_albedo.reflection + mix_value * fg_albedo.reflection,
            bg_mix * bg_albedo.absorption_r + mix_value * fg_albedo.absorption_r,
            bg_mix * bg_albedo.transmission + mix_value * fg_albedo.transmission,
            bg_mix * bg_albedo.absorption_t + mix_value * fg_albedo.absorption_t
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
    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    public override float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf(wi, wo, sg, bc).pdf;
    }

    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        BSDFEval result1 = in1.eval_with_pdf(wi, wo, sg, bc);
        BSDFEval result2 = in2.eval_with_pdf(wi, wo, sg, bc);
        float p = p_in1(wi);
        return BSDFEval(result1.value + result2.value, p * result1.pdf + (1.0 - p) * result2.pdf);
    }
    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};

        float p = p_in1(wi);
        MxBranchChoice choice = mx_choose_branch(sg, p);
        if (choice.pdf <= 0.0)
            return false;
        bool valid = choice.first ? in1.sample(wi, sample, sg, bc) : in2.sample(wi, sample, sg, bc);
        if (!valid)
            return false;
        if (sample.has_flag(BSDFFlags::delta)) {
            sample.pdf = choice.pdf * sample.pdf;
            sample.weight = sample.weight / choice.pdf;
            return true;
        }
        const BSDFEval eval_result = eval_with_pdf(wi, sample.wo, sg, bc);
        sample.pdf = eval_result.pdf;
        if (sample.pdf <= 0.0)
            return false;
        sample.weight = eval_result.value / sample.pdf;
        return true;
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        AlbedoContributions albedo1 = in1.eval_albedo(wi);
        AlbedoContributions albedo2 = in2.eval_albedo(wi);
        return from_scattering_albedo(
            albedo1.reflection + albedo2.reflection,
            albedo1.transmission + albedo2.transmission
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

} // namespace mtlx
