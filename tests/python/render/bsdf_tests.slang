// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
__exported import falcor2.render;
__exported import falcor2.utils;
__exported import falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.mx139;

/// BSDF test utilities for chi-square and other statistical tests.
namespace bsdf_tests {

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

public float4 test_mx139_beer_delta_transmission(mx139::MxBeerBSDF bsdf, float3 wi)
{
    UniformSampleGenerator sg = UniformSampleGenerator(1);
    BSDFSample sample = {};
    BSDFContext bc = {};

    bool valid = bsdf.sample(wi, sample, sg, bc);
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
        bool valid = bsdf.sample(wi, sample, sg, bc);
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
/// Counter indices: [0]=valid, [1]=failed, [2]=delta, [3]=negative
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

        if (!bsdf.sample(wi, sample, sg, bc)) {
            counters.add(1, 1); // failed
            continue;
        }

        // Skip delta lobes (pdf=0 by convention, can't verify eval/pdf ratio).
        if (sample.has_flag(BSDFFlags::delta)) {
            counters.add(2, 1); // delta
            continue;
        }

        counters.add(0, 1); // valid

        // Sanity: check for negative values.
        if (sample.pdf < 0 || any(sample.weight < 0)) {
            counters.add(3, 1); // negative
            continue;
        }

        // Verify pdf consistency: eval_pdf(wi, wo) should match pdf from sample().
        float eval_pdf_value = bsdf.eval_pdf(wi, sample.wo, bc);
        float pdf_denom = max(abs(sample.pdf), 1e-8);
        float pdf_err = abs(eval_pdf_value - sample.pdf) / pdf_denom;
        max_pdf_err = max(max_pdf_err, pdf_err);

        // Verify weight consistency: eval(wi, wo) / eval_pdf(wi, wo) ~= weight.
        if (eval_pdf_value > 1e-8) {
            float3 eval_value = bsdf.eval(wi, sample.wo, sg, bc);
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

        if (!bsdf.sample(wi, sample, sg, bc))
            continue;

        // Only test non-delta reflection (both directions in upper hemisphere).
        if (sample.has_flag(BSDFFlags::delta) || sample.wo.z <= BSDF_MIN_COS_THETA) {
            counters.add(1, 1); // skipped
            continue;
        }

        counters.add(0, 1); // valid

        // eval(wi, wo) = f(wi, wo) * |wo.z|
        // eval(wo, wi) = f(wo, wi) * |wi.z|
        // Reciprocity: f(wi, wo) = f(wo, wi)
        // => eval(wi, wo) * |wi.z| = eval(wo, wi) * |wo.z|
        float3 eval_fwd = bsdf.eval(wi, sample.wo, sg, bc);
        float3 eval_rev = bsdf.eval(sample.wo, wi, sg, bc);

        float3 lhs = eval_fwd * abs(wi.z);
        float3 rhs = eval_rev * abs(sample.wo.z);

        for (int c = 0; c < 3; c++) {
            float denom = max(max(abs(lhs[c]), abs(rhs[c])), 1e-8);
            float err = abs(lhs[c] - rhs[c]) / denom;
            max_err = max(max_err, err);
        }
    }

    errors.store(tid, max_err);
}

} // namespace bsdf_tests
