// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
__exported import falcor2.render;
__exported import falcor2.utils;
__exported import falcor2.render.materials.mtlx.mtlx;

/// BSDF test utilities for chi-square and other statistical tests.
namespace bsdf_tests {

// Reconstruct at each incident direction: lobe-selection probabilities depend on wi.
// Keep this flat for SlangPy's existing dictionary fixtures. The production
// GLTFBSDFData uses conditional feature groups instead.
public struct TestGLTFBSDFData
{
    public float3 base_color = {};
    public float metallic = {};
    public float roughness = {};
    public float anisotropy_strength = {};
    public float2 anisotropy_direction = float2(1.0, 0.0);
    public float diffuse_transmission = {};
    public float3 diffuse_transmission_color = float3(1.0);
    public float3 sheen_color = {};
    public float sheen_roughness = {};
    public float iridescence = {};
    public float iridescence_ior = 1.3;
    public float iridescence_thickness = 400.0;
    public float3 dielectric_f0 = float3(0.04);
    public float dielectric_f90 = 1.0;
    public float transmission = {};
    public float ior = 1.5;
    public float eta = 1.0;
    public bool thin_walled = true;
};

public struct TestGLTFBSDF : IBSDF
{
    public TestGLTFBSDFData data;

    private uint runtime_features()
    {
        uint mask = 0u;
        if (data.anisotropy_strength > 0.0)
            mask |= gltfmat::FeatureBits::ANISOTROPY;
        if (data.diffuse_transmission > 0.0)
            mask |= gltfmat::FeatureBits::DIFFUSE_TRANSMISSION;
        if (any(data.sheen_color > float3(0.0)))
            mask |= gltfmat::FeatureBits::SHEEN;
        if (data.iridescence > 0.0)
            mask |= gltfmat::FeatureBits::IRIDESCENCE;
        if (data.transmission > 0.0)
            mask |= gltfmat::FeatureBits::TRANSMISSION;
        return mask;
    }

    private GLTFBSDFData<Features> prepare_data<let Features : uint>()
    {
        GLTFBSDFData<Features> prepared = {};
        prepared.base_color = data.base_color;
        prepared.metallic = data.metallic;
        prepared.roughness = data.roughness;
        prepared.dielectric_f0 = data.dielectric_f0;
        prepared.dielectric_f90 = data.dielectric_f90;
        prepared.ior = data.ior;
        prepared.eta = data.eta;
        prepared.thin_walled = data.thin_walled;
        if (let available = prepared.anisotropy.get())
            prepared.anisotropy.set(GLTFAnisotropyInputs(data.anisotropy_strength, data.anisotropy_direction));
        if (let available = prepared.diffuse_transmission.get())
            prepared.diffuse_transmission.set(
                GLTFDiffuseTransmissionInputs(data.diffuse_transmission, data.diffuse_transmission_color)
            );
        if (let available = prepared.sheen.get())
            prepared.sheen.set(GLTFSheenInputs(data.sheen_color, data.sheen_roughness));
        if (let available = prepared.iridescence.get())
            prepared.iridescence.set(
                GLTFIridescenceInputs(data.iridescence, data.iridescence_ior, data.iridescence_thickness)
            );
        if (let available = prepared.transmission.get())
            prepared.transmission.set(GLTFTransmissionInputs(data.transmission));
        return prepared;
    }

    private GLTFBSDF<Features> make_bsdf<let Features : uint>(float3 wi)
    {
        return GLTFBSDF<Features>(wi, prepare_data<Features>(), runtime_features());
    }

    public float4 compare_exact_flexible<let Features : uint>(float3 wi, float3 wo)
    {
        GLTFBSDF<Features> exact = make_bsdf<Features>(wi);
        GLTFBSDF<gltfmat::FeatureBits::ALL_MR> flexible = make_bsdf<gltfmat::FeatureBits::ALL_MR>(wi);
        BSDFEval a = exact.eval_with_pdf(wi, wo, BSDFContext());
        BSDFEval b = flexible.eval_with_pdf(wi, wo, BSDFContext());
        AlbedoContributions aa = exact.eval_albedo(wi);
        AlbedoContributions bb = flexible.eval_albedo(wi);
        return float4(
            length(a.value - b.value),
            abs(a.pdf - b.pdf),
            length(aa.reflection - bb.reflection) + length(aa.transmission - bb.transmission),
            length(aa.absorption_r - bb.absorption_r) + length(aa.absorption_t - bb.absorption_t)
        );
    }

    public BSDFEval eval_with_pdf(float3 wi, float3 wo, BSDFContext bc)
    {
        return make_bsdf<gltfmat::FeatureBits::ALL_MR>(wi).eval_with_pdf(wi, wo, bc);
    }
    public bool sample(float3 wi, const BSDFSampleInput input, const BSDFContext bc, out BSDFSample result)
    {
        return make_bsdf<gltfmat::FeatureBits::ALL_MR>(wi).sample(wi, input, bc, result);
    }
    public AlbedoContributions eval_albedo(float3 wi)
    {
        return make_bsdf<gltfmat::FeatureBits::ALL_MR>(wi).eval_albedo(wi);
    }
    public RoughnessInformation eval_roughness(float3 wi)
    {
        return make_bsdf<gltfmat::FeatureBits::ALL_MR>(wi).eval_roughness(wi);
    }
    public property float3 ior_as_reflectance
    {
        get {
            return make_bsdf<gltfmat::FeatureBits::ALL_MR>(float3(0.0, 0.0, 1.0)).ior_as_reflectance;
        }
    }
};

public float4 test_gltf_exact_flexible<let Features : uint>(TestGLTFBSDFData data, float3 wi, float3 wo)
{
    TestGLTFBSDF test = {};
    test.data = data;
    return test.compare_exact_flexible<Features>(wi, wo);
}

public struct TestGLTFSpecGlossBSDF : IBSDF
{
    public GLTFSpecGlossBSDFData data;
    public BSDFEval eval_with_pdf(float3 wi, float3 wo, BSDFContext bc)
    {
        return GLTFSpecGlossBSDF(wi, data).eval_with_pdf(wi, wo, bc);
    }
    public bool sample(float3 wi, const BSDFSampleInput input, const BSDFContext bc, out BSDFSample result)
    {
        return GLTFSpecGlossBSDF(wi, data).sample(wi, input, bc, result);
    }
    public AlbedoContributions eval_albedo(float3 wi) { return GLTFSpecGlossBSDF(wi, data).eval_albedo(wi); }
    public RoughnessInformation eval_roughness(float3 wi) { return GLTFSpecGlossBSDF(wi, data).eval_roughness(wi); }
    public property float3 ior_as_reflectance
    {
        get {
            return GLTFSpecGlossBSDF(float3(0.0, 0.0, 1.0), data).ior_as_reflectance;
        }
    }
};

public struct FixedSampleGenerator : ISampleGenerator
{
    uint3 samples;
    uint index;

    public __init(uint seed, uint scrambling_seed = 0)
    {
        samples = uint3(seed);
        index = 0;
    }

    public __init(uint2 pixel, uint sample_index)
    {
        samples = uint3(pixel.x, pixel.y, sample_index);
        index = 0;
    }

    public __init(float3 sample)
    {
        samples.x = uint(sample.x * 0x1p24) << 8;
        samples.y = uint(sample.y * 0x1p24) << 8;
        samples.z = uint(sample.z * 0x1p24) << 8;
        index = 0;
    }

    [mutating]
    public uint next_uint()
    {
        uint result = samples[min(index, 2)];
        index++;
        return result;
    }
}

public float4 test_anisotropic_ggx_policy()
{
    AnisotropicGGX alias_zero = AnisotropicGGX(float2(0.0));
    AnisotropicGGXPolicy<1000, true> clamped_zero = AnisotropicGGXPolicy<1000, true>(float2(0.0));
    AnisotropicGGXPolicy<100, false> wide_cutoff = AnisotropicGGXPolicy<100, false>(float2(0.005));

    return float4(
        alias_zero.is_singular() ? 1.0 : 0.0,
        clamped_zero.is_singular() ? 1.0 : 0.0,
        clamped_zero.alpha.x,
        wide_cutoff.is_singular() ? 1.0 : 0.0
    );
}

public float probe_anisotropic_ggx_ndf(float2 alpha, float3 h)
{
    return AnisotropicGGXPolicy<10000, true>(alpha).eval_ndf(h);
}

public float4 test_diffuse_specular_albedo(DiffuseSpecularBRDF bsdf, float3 wi)
{
    AlbedoContributions albedo = bsdf.eval_albedo(wi);

    float x = saturate(wi.z);
    float one_minus_x = 1.0 - x;
    float one_minus_x2 = one_minus_x * one_minus_x;
    float fit0 = 0.97619 - 0.488095 * one_minus_x2 * one_minus_x2 * one_minus_x;
    float fit1 = 1.55754 + (-2.02221 + (2.56283 - 1.06244 * x) * x) * x;
    float diffuse_albedo = lerp(fit0, fit1, bsdf.roughness);
    float3 specular_albedo
        = mtlx::ggx_directional_albedo_analytic(x, bsdf.roughness * bsdf.roughness, bsdf.specular, float3(1.0));
    float3 expected_reflection = saturate(bsdf.diffuse * diffuse_albedo + specular_albedo);

    float3 total = albedo.reflection + albedo.absorption_r + albedo.transmission + albedo.absorption_t;
    return float4(
        length(albedo.reflection - expected_reflection),
        length(total - 1.0),
        length(albedo.transmission),
        length(albedo.absorption_t)
    );
}

public float4 test_mtlx_mix_albedo_contributions()
{
    mtlx::MxTranslucentBSDF bg = mtlx::MxTranslucentBSDF(float3(0.2));
    mtlx::MxTranslucentBSDF fg = mtlx::MxTranslucentBSDF(float3(0.8));
    mtlx::MxMixBSDF<mtlx::MxTranslucentBSDF, mtlx::MxTranslucentBSDF> bsdf
        = mtlx::MxMixBSDF<mtlx::MxTranslucentBSDF, mtlx::MxTranslucentBSDF>(fg, bg, 0.25);
    AlbedoContributions albedo = bsdf.eval_albedo(float3(0.0, 0.0, 1.0));
    return float4(albedo.reflection.x, albedo.absorption_r.x, albedo.transmission.x, albedo.absorption_t.x);
}

public float4 test_mtlx_beer_delta_transmission(mtlx::MxBeerBSDF bsdf, float3 wi)
{
    UniformSampleGenerator sg = UniformSampleGenerator(1);
    BSDFSample sample = {};
    BSDFContext bc = {};

    bool valid = bsdf.sample(wi, make_bsdf_sample_input(sg), bc, sample);
    bool delta_transmission = sample.has_flag(BSDFFlags::delta_transmission);
    AlbedoContributions albedo = bsdf.eval_albedo(wi);
    float wo_error = length(sample.wo - float3(-wi.x, -wi.y, -wi.z));

    return float4(
        valid && delta_transmission ? 1.0 : 0.0,
        wo_error + abs(sample.pdf),
        sample.weight.x,
        albedo.transmission.z
    );
}

public float4 test_controlled_bsdf_sample<B : IBSDF>(B bsdf, float3 wi, BSDFSampleInput input)
{
    BSDFSample sample = {};
    BSDFContext bc = {};

    bool valid = bsdf.sample(wi, input, bc, sample);
    float eval_pdf = valid ? bsdf.eval_pdf(wi, sample.wo, bc) : 0.0;
    return float4(valid ? 1.0 : 0.0, sample.wo.z, sample.pdf, eval_pdf);
}

// Check the sampling mask independently of the physical scatter mode and compensation.
public float4 test_mtlx_lobe_sampling_hint<B : IBSDF>(B bsdf, float3 wi, float3 random, uint hint, uint check)
{
    const BSDFSampleInput input = { random.z, random.xy, 0u };
    BSDFContext bc = BSDFContext(1.0, 1.5, BSDFFlags(hint));
    BSDFContext unrestricted = BSDFContext(1.0, 1.5);
    BSDFSample sample = {};
    bool valid = bsdf.sample(wi, input, bc, sample);
    if (check == 0)
        return float4(valid ? 1.0 : 0.0, sample.wo.z, sample.pdf, bsdf.eval_pdf(wi, sample.wo, bc));
    if (check == 1) {
        float3 value = bsdf.eval(wi, sample.wo, unrestricted);
        float3 hinted_value = bsdf.eval(wi, sample.wo, bc);
        float scale = max(max(value.x, value.y), max(value.z, 1e-6));
        return float4((sample.weight * sample.pdf - value) / scale, length(hinted_value - value) / scale);
    }
    if (check == 2) {
        float3 reflected = reflect(-wi, float3(0.0, 0.0, 1.0));
        float3 transmitted = refract(-wi, float3(0.0, 0.0, 1.0), 1.0 / 1.5);
        return float4(
            bsdf.eval_pdf(wi, reflected, bc),
            bsdf.eval_pdf(wi, transmitted, bc),
            bsdf.eval_pdf(wi, reflected, unrestricted),
            bsdf.eval_pdf(wi, transmitted, unrestricted)
        );
    }
    if (check == 4 || check == 5) {
        float3 wo = check == 4 ? reflect(-wi, float3(0.0, 0.0, 1.0)) : refract(-wi, float3(0.0, 0.0, 1.0), 1.0 / 1.5);
        BSDFEval baseline = bsdf.eval_with_pdf(wi, wo, unrestricted);
        BSDFEval hinted = bsdf.eval_with_pdf(wi, wo, bc);
        float scale = max(length(baseline.value), 1e-6);
        return float4((hinted.value - baseline.value) / scale, hinted.pdf - bsdf.eval_pdf(wi, wo, bc));
    }
    // Failed samples must not expose stale weights, flags, or density.
    return float4(length(sample.weight), sample.pdf, float(uint(sample.flags)), length(sample.wo));
}

public float4 test_standard_bsdf_diffuse_transmission_support(StandardBSDFData data)
{
    float3 wi = float3(0.0, 0.0, 1.0);
    float3 wo = float3(0.0, 0.0, -1.0);
    StandardBSDF bsdf = StandardBSDF(wi, data);
    FixedSampleGenerator sg = FixedSampleGenerator(float3(0.5));
    BSDFContext bc = {};

    float3 value = bsdf.eval(wi, wo, bc);
    BSDFSample sample = {};
    bool valid
        = bsdf.sample(wi, make_bsdf_sample_input(sg), bc, sample) && sample.has_flag(BSDFFlags::diffuse_transmission);
    bool has_lobe = bsdf.get_lobe_types() & BSDFFlags::diffuse_transmission;
    AlbedoContributions albedo = bsdf.eval_albedo(wi);

    return float4(value.x, sample.weight.x, valid && has_lobe ? 1.0 : 0.0, albedo.transmission.x);
}

public float4 test_specular_microfacet_smooth_transmission(SpecularMicrofacetBSDF bsdf)
{
    FixedSampleGenerator sg = FixedSampleGenerator(float3(0.5));
    BSDFSample sample = {};
    BSDFContext bc = {};

    bool valid = bsdf.sample(float3(0.0, 0.0, 1.0), make_bsdf_sample_input(sg), bc, sample);
    bool delta_transmission = sample.has_flag(BSDFFlags::delta_transmission);
    return float4(valid && delta_transmission ? 1.0 : 0.0, sample.pdf, sample.weight.x, sample.weight.z);
}

public float4 test_specular_microfacet_brdf_albedo(SpecularMicrofacetBRDF bsdf, uint channel)
{
    AlbedoContributions albedo = bsdf.eval_albedo(float3(0.0, 0.0, 1.0));
    return float4(
        albedo.reflection[channel],
        albedo.absorption_r[channel],
        albedo.transmission[channel],
        albedo.absorption_t[channel]
    );
}

public float4 test_specular_microfacet_bsdf_albedo(SpecularMicrofacetBSDF bsdf, uint channel)
{
    AlbedoContributions albedo = bsdf.eval_albedo(float3(0.0, 0.0, 1.0));
    return float4(
        albedo.reflection[channel],
        albedo.absorption_r[channel],
        albedo.transmission[channel],
        albedo.absorption_t[channel]
    );
}

public float4 test_pbrt_dielectric_event_support(PBRTDielectricBSDF bsdf)
{
    FixedSampleGenerator sg = FixedSampleGenerator(float3(0.5));
    BSDFSample sample = {};

    const float3 wi_tir = normalize(float3(sqrt(0.75), 0.0, 0.5));
    BSDFContext transmission_only = BSDFContext(1.5, 1.0, BSDFFlags::transmission);
    bool accepted_tir = bsdf.sample(wi_tir, make_bsdf_sample_input(sg), transmission_only, sample);

    BSDFContext reflection_only = BSDFContext(1.0, 1.0, BSDFFlags::reflection);
    bool accepted_zero_reflection
        = bsdf.sample(float3(0.0, 0.0, 1.0), make_bsdf_sample_input(sg), reflection_only, sample);

    AlbedoContributions albedo = bsdf.eval_albedo(wi_tir);
    return float4(
        accepted_tir ? 1.0 : 0.0,
        accepted_zero_reflection ? 1.0 : 0.0,
        albedo.reflection.x,
        albedo.transmission.x
    );
}

public float4 test_pbrt_dielectric_transmission_scaling(PBRTDielectricBSDF smooth_bsdf, PBRTDielectricBSDF rough_bsdf)
{
    const float3 wi = float3(0.0, 0.0, 1.0);
    BSDFContext bc = BSDFContext(1.0, 1.5, BSDFFlags::all);

    FixedSampleGenerator smooth_sg = FixedSampleGenerator(float3(0.5));
    BSDFSample smooth_sample = {};
    bool smooth_valid = smooth_bsdf.sample(wi, make_bsdf_sample_input(smooth_sg), bc, smooth_sample)
        && smooth_sample.has_flag(BSDFFlags::delta_transmission);

    FixedSampleGenerator rough_sg = FixedSampleGenerator(float3(0.5));
    BSDFSample rough_sample = {};
    bool rough_valid = rough_bsdf.sample(wi, make_bsdf_sample_input(rough_sg), bc, rough_sample)
        && rough_sample.has_flag(BSDFFlags::glossy_transmission);
    float geometry_weight = rough_bsdf.D.G2(rough_sample.wo, wi) / rough_bsdf.D.G1(wi);
    float sample_scale = rough_sample.weight.x / geometry_weight;
    float eval_scale = rough_bsdf.eval(wi, rough_sample.wo, bc).x / rough_sample.pdf / geometry_weight;

    return float4(smooth_valid && rough_valid ? 1.0 : 0.0, smooth_sample.weight.x, sample_scale, eval_scale);
}

float4 test_mtlx_retroreflection_eval_pdf<B : IBSDF>(B regular, B retroreflective)
{
    const float3 wi = normalize(float3(0.3, 0.2, 0.9));
    const float3 wo = normalize(float3(-0.1, 0.4, 0.9));
    const float3 reference_wi = mtlx::mx_retroreflect(wi);
    BSDFContext bc = {};

    const float3 regular_eval = regular.eval(reference_wi, wo, bc);
    const float3 retroreflective_eval = retroreflective.eval(wi, wo, bc);
    const float regular_pdf = regular.eval_pdf(reference_wi, wo, bc);
    const float retroreflective_pdf = retroreflective.eval_pdf(wi, wo, bc);

    return float4(
        mtlx::mx_hmax(abs(regular_eval - retroreflective_eval)),
        abs(regular_pdf - retroreflective_pdf),
        regular_pdf,
        retroreflective_pdf
    );
}

float4 test_mtlx_retroreflection_sample<B : IBSDF>(B regular, B retroreflective)
{
    const float3 wi = normalize(float3(0.3, 0.2, 0.9));
    BSDFContext bc = {};
    float direction_error = 0.0;
    float pdf_error = 0.0;
    float weight_and_eval_error = 0.0;
    uint found_events = 0;

    for (uint seed = 1; seed <= 256; ++seed) {
        UniformSampleGenerator regular_sg = UniformSampleGenerator(seed);
        UniformSampleGenerator retroreflective_sg = UniformSampleGenerator(seed);
        BSDFSample regular_sample = {};
        BSDFSample retroreflective_sample = {};
        const bool regular_valid = regular.sample(wi, make_bsdf_sample_input(regular_sg), bc, regular_sample);
        const bool retroreflective_valid
            = retroreflective.sample(wi, make_bsdf_sample_input(retroreflective_sg), bc, retroreflective_sample);
        if (regular_valid != retroreflective_valid) {
            direction_error = 1.0;
            continue;
        }
        if (!regular_valid)
            continue;

        const bool reflection = wi.z * regular_sample.wo.z > 0.0;
        found_events |= reflection ? 1 : 2;
        const float3 expected_wo = reflection ? mtlx::mx_retroreflect(regular_sample.wo) : regular_sample.wo;
        direction_error = max(direction_error, length(expected_wo - retroreflective_sample.wo));
        pdf_error = max(pdf_error, abs(regular_sample.pdf - retroreflective_sample.pdf));
        weight_and_eval_error
            = max(weight_and_eval_error, mtlx::mx_hmax(abs(regular_sample.weight - retroreflective_sample.weight)));

        const float3 regular_eval = regular.eval(wi, regular_sample.wo, bc);
        const float3 retroreflective_eval = retroreflective.eval(wi, retroreflective_sample.wo, bc);
        weight_and_eval_error = max(weight_and_eval_error, mtlx::mx_hmax(abs(regular_eval - retroreflective_eval)));
        pdf_error = max(
            pdf_error,
            abs(regular.eval_pdf(wi, regular_sample.wo, bc)
                - retroreflective.eval_pdf(wi, retroreflective_sample.wo, bc))
        );
    }

    return float4(direction_error, pdf_error, weight_and_eval_error, float(found_events));
}

mtlx::MxConductorBSDF make_test_mtlx_conductor(bool retroreflective)
{
    return mtlx::create_mx_conductor(
        float3(0.2, 0.5, 1.0),
        float3(3.0, 2.0, 1.5),
        float2(0.25, 0.45),
        retroreflective,
        0.0,
        1.5,
        float3(0.0, 0.0, 1.0),
        float3(1.0, 0.0, 0.0),
        0
    );
}

mtlx::MxDielectricBSDF make_test_mtlx_dielectric(bool retroreflective)
{
    return mtlx::create_mx_dielectric(
        float3(1.0),
        3.0,
        float2(0.25, 0.45),
        retroreflective,
        0.0,
        1.5,
        float3(0.0, 0.0, 1.0),
        float3(1.0, 0.0, 0.0),
        0,
        2,
        true
    );
}

mtlx::MxGeneralizedSchlickBSDF make_test_mtlx_generalized_schlick(bool retroreflective)
{
    return mtlx::create_mx_generalized_schlick(
        float3(0.5),
        float3(0.7),
        float3(1.0),
        5.0,
        float2(0.25, 0.45),
        retroreflective,
        0.0,
        1.5,
        float3(0.0, 0.0, 1.0),
        float3(1.0, 0.0, 0.0),
        0,
        2,
        true
    );
}

public float4 test_mtlx_conductor_retroreflection_eval_pdf()
{
    return test_mtlx_retroreflection_eval_pdf(make_test_mtlx_conductor(false), make_test_mtlx_conductor(true));
}

public float4 test_mtlx_conductor_retroreflection_sample()
{
    return test_mtlx_retroreflection_sample(make_test_mtlx_conductor(false), make_test_mtlx_conductor(true));
}

public float4 test_mtlx_dielectric_retroreflection_eval_pdf()
{
    return test_mtlx_retroreflection_eval_pdf(make_test_mtlx_dielectric(false), make_test_mtlx_dielectric(true));
}

public float4 test_mtlx_dielectric_retroreflection_sample()
{
    return test_mtlx_retroreflection_sample(make_test_mtlx_dielectric(false), make_test_mtlx_dielectric(true));
}

public float4 test_mtlx_generalized_schlick_retroreflection_eval_pdf()
{
    return test_mtlx_retroreflection_eval_pdf(
        make_test_mtlx_generalized_schlick(false),
        make_test_mtlx_generalized_schlick(true)
    );
}

public float4 test_mtlx_generalized_schlick_retroreflection_sample()
{
    return test_mtlx_retroreflection_sample(
        make_test_mtlx_generalized_schlick(false),
        make_test_mtlx_generalized_schlick(true)
    );
}

// ============================================================================
// Chi-square test target
// ============================================================================

/// Chi-square test target wrapping any BSDF implementation.
///
/// This adapts the IBSDF interface to the chi2::ITarget interface,
/// using SphericalDomain for the parameter space.
/// Tests that sample() produces directions distributed according to eval_pdf().
///
/// The domain bounds are configurable to support testing reflection-only BSDFs
/// (upper hemisphere) or full-sphere BSDFs (reflection + transmission).
/// Using hemisphere bounds avoids binning artifacts at the equatorial
/// discontinuity where eval_pdf transitions from >0 to 0.
///
/// Note: This only tests continuous (non-delta) lobes. BSDFs with purely
/// delta distributions (e.g., perfect mirrors) cannot be tested this way.
public struct BSDFTarget<B : IBSDF> : chi2::ITarget
{
    public B bsdf;
    public float3 wi;
    public BSDFContext bc;

    typedef chi2::SphericalDomain Domain;

    public Domain get_domain() { return chi2::SphericalDomain(); }

    public bool sample<SG : ISampleGenerator>(inout SG sg, out float3 p)
    {
        BSDFSample sample;
        bool valid = bsdf.sample(wi, make_bsdf_sample_input(sg), bc, sample);
        p = sample.wo;
        return valid;
    }

    public float pdf(float3 wo) { return bsdf.eval_pdf(wi, wo, bc); }
};

// ============================================================================
// Eval/sample consistency test kernel
// ============================================================================

/// GPU kernel to test eval/sample consistency of a BSDF.
///
/// For each non-delta sample drawn from bsdf.sample(), verifies:
/// 1. eval(wi, wo) / eval_pdf(wi, wo) ~= weight (weight consistency)
/// 2. eval_pdf(wi, wo) ~= pdf from sample() (pdf consistency)
///
/// Results are written per-thread (max errors). The Python side reduces
/// across threads by taking the maximum.
///
/// Counter indices: [0]=valid, [1]=failed, [2]=delta, [3]=invalid
public void test_eval_sample_consistency<B : IBSDF>(
    uint tid,
    uint sample_count,
    uint samples_per_thread,
    uint seed,
    B bsdf,
    float3 wi,
    BSDFContext bc,
    WTensor<float, 1> weight_errors,
    WTensor<float, 1> pdf_errors,
    AtomicTensor<uint, 1> counters
)
{
    uint2 v = block_cipher_tea(seed, tid);
    UniformSampleGenerator sg = UniformSampleGenerator(v.x);

    float max_weight_err = 0.0;
    float max_pdf_err = 0.0;

    for (uint i = 0; i < samples_per_thread; ++i) {
        uint sample_index = tid * samples_per_thread + i;
        if (sample_index >= sample_count)
            break;

        BSDFSample sample = {};

        if (!bsdf.sample(wi, make_bsdf_sample_input(sg), bc, sample)) {
            counters.add(1, 1); // failed
            continue;
        }

        // Sanity: check the complete sample payload before handling either measure.
        if (sample.pdf < 0 || any(sample.weight < 0) || isnan(sample.pdf) || isinf(sample.pdf)
            || any(isnan(sample.weight)) || any(isinf(sample.weight)) || any(isnan(sample.wo))
            || any(isinf(sample.wo))) {
            counters.add(3, 1); // invalid
            continue;
        }

        // Skip delta lobes after checking their zero solid-angle PDF contract.
        if (sample.has_flag(BSDFFlags::delta)) {
            if (sample.pdf == 0.0)
                counters.add(2, 1); // delta
            else
                counters.add(3, 1); // invalid
            continue;
        }

        counters.add(0, 1); // valid

        // Verify pdf consistency: eval_pdf(wi, wo) should match pdf from sample().
        float eval_pdf_value = bsdf.eval_pdf(wi, sample.wo, bc);
        float pdf_denom = max(abs(sample.pdf), 1e-8);
        float pdf_err = abs(eval_pdf_value - sample.pdf) / pdf_denom;
        max_pdf_err = max(max_pdf_err, pdf_err);

        // Verify weight consistency: eval(wi, wo) / eval_pdf(wi, wo) ~= weight.
        if (eval_pdf_value > 1e-8) {
            float3 eval_value = bsdf.eval(wi, sample.wo, bc);
            float3 computed_weight = eval_value / eval_pdf_value;

            // Per-channel relative error.
            for (int c = 0; c < 3; c++) {
                float denom = max(abs(sample.weight[c]), 1e-8);
                float err = abs(computed_weight[c] - sample.weight[c]) / denom;
                max_weight_err = max(max_weight_err, err);
            }
        }
    }

    weight_errors.store(tid, max_weight_err);
    pdf_errors.store(tid, max_pdf_err);
}

// ============================================================================
// Reciprocity test kernel
// ============================================================================

/// GPU kernel to test BSDF reciprocity (Helmholtz reciprocity).
///
/// For non-delta reflection lobes, verifies:
///   eval(wi, wo) * |wi.z| ~= eval(wo, wi) * |wo.z|
///
/// This follows from f(wi, wo) = f(wo, wi) and the convention
/// that eval returns f * |cos(theta_o)|.
///
/// Only tests directions where both wi and wo are in the positive hemisphere,
/// since IBSDF requires wi to be in the positive hemisphere.
///
/// Counter indices: [0]=valid, [1]=skipped
public void test_reciprocity<B : IBSDF>(
    uint tid,
    uint sample_count,
    uint samples_per_thread,
    uint seed,
    B bsdf,
    float3 wi,
    BSDFContext bc,
    WTensor<float, 1> errors,
    AtomicTensor<uint, 1> counters
)
{
    uint2 v = block_cipher_tea(seed, tid);
    UniformSampleGenerator sg = UniformSampleGenerator(v.x);

    float max_err = 0.0;

    for (uint i = 0; i < samples_per_thread; ++i) {
        uint sample_index = tid * samples_per_thread + i;
        if (sample_index >= sample_count)
            break;

        BSDFSample sample = {};

        if (!bsdf.sample(wi, make_bsdf_sample_input(sg), bc, sample))
            continue;

        // Only test non-delta reflection (both directions in upper hemisphere).
        if (sample.has_flag(BSDFFlags::delta) || sample.wo.z <= BSDF_MIN_COS_THETA) {
            counters.add(1, 1); // skipped
            continue;
        }

        counters.add(0, 1); // valid

        // Reflection arithmetic can leave wo slightly non-unit. Normalize before
        // swapping incident/outgoing directions; near the TIR boundary, that
        // length error can otherwise dominate the Fresnel reciprocity check.
        const float3 wo = normalize(sample.wo);

        // eval(wi, wo) = f(wi, wo) * |wo.z|
        // eval(wo, wi) = f(wo, wi) * |wi.z|
        // Reciprocity: f(wi, wo) = f(wo, wi)
        // => eval(wi, wo) * |wi.z| = eval(wo, wi) * |wo.z|
        float3 eval_fwd = bsdf.eval(wi, wo, bc);
        float3 eval_rev = bsdf.eval(wo, wi, bc);

        float3 lhs = eval_fwd * abs(wi.z);
        float3 rhs = eval_rev * abs(wo.z);

        for (int c = 0; c < 3; c++) {
            float denom = max(max(abs(lhs[c]), abs(rhs[c])), 1e-8);
            float err = abs(lhs[c] - rhs[c]) / denom;
            max_err = max(max_err, err);
        }
    }

    errors.store(tid, max_err);
}

} // namespace bsdf_tests
