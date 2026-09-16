// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
__exported import falcor2.render;
__exported import falcor2.utils;

namespace phase_function_tests {

/// Non-empty adapter for CUDA, which cannot currently dispatch a reflected
/// generic parameter containing the fieldless IsotropicPhaseFunction type.
/// TODO: Remove this adapter when CUDA supports reflected zero-sized generic fields.
public struct TestIsotropicPhaseFunction : IPhaseFunction
{
    public float dummy;

    public float3 eval(const float3 wi, const float3 wo) { return IsotropicPhaseFunction().eval(wi, wo); }

    public bool sample<SG : ISampleGenerator>(const float3 wi, inout SG sg, out PhaseFunctionSample sample)
    {
        return IsotropicPhaseFunction().sample(wi, sg, sample);
    }

    public float eval_pdf(const float3 wi, const float3 wo) { return IsotropicPhaseFunction().eval_pdf(wi, wo); }
};

/// Adapts a phase function to the chi-square framework's spherical domain.
public struct PhaseFunctionTarget<P : IPhaseFunction> : chi2::ITarget
{
    public P phase_function;
    public float3 wi;

    typedef chi2::SphericalDomain Domain;

    public Domain get_domain() { return chi2::SphericalDomain(); }

    public bool sample<SG : ISampleGenerator>(inout SG sg, out float3 p)
    {
        PhaseFunctionSample sample = {};
        bool valid = phase_function.sample(wi, sg, sample);
        p = sample.wo_ws;
        return valid;
    }

    public float pdf(float3 wo) { return phase_function.eval_pdf(wi, wo); }
};

public float4 test_null_phase_function_sample(float3 wi)
{
    UniformSampleGenerator sg = UniformSampleGenerator(1);
    PhaseFunctionSample sample;
    bool valid = NullPhaseFunction().sample(wi, sg, sample);
    return float4(valid ? 1.0 : 0.0, sample.pdf, sample.weight.x, length(sample.wo_ws));
}

/// Checks the deterministic contract of PhaseFunctionSample independently of
/// the distribution-level chi-square test.
public void test_eval_sample_consistency<P : IPhaseFunction>(
    uint tid,
    uint sample_count,
    uint samples_per_thread,
    uint seed,
    P phase_function,
    float3 wi,
    WTensor<float, 1> weight_errors,
    WTensor<float, 1> pdf_errors,
    WTensor<float, 1> direction_errors,
    AtomicTensor<uint, 1> counters
)
{
    uint2 v = block_cipher_tea(seed, tid);
    UniformSampleGenerator sg = UniformSampleGenerator(v.x);

    float max_weight_error = 0.0;
    float max_pdf_error = 0.0;
    float max_direction_error = 0.0;

    for (uint i = 0; i < samples_per_thread; ++i) {
        uint sample_index = tid * samples_per_thread + i;
        if (sample_index >= sample_count)
            break;

        PhaseFunctionSample sample = {};
        if (!phase_function.sample(wi, sg, sample)) {
            counters.add(1, 1);
            continue;
        }

        counters.add(0, 1);

        if (sample.pdf < 0.0 || any(sample.weight < 0.0) || !all(isfinite(sample.wo_ws))) {
            counters.add(2, 1);
            continue;
        }

        float eval_pdf_value = phase_function.eval_pdf(wi, sample.wo_ws);
        float pdf_denom = max(max(abs(sample.pdf), abs(eval_pdf_value)), 1e-8);
        max_pdf_error = max(max_pdf_error, abs(eval_pdf_value - sample.pdf) / pdf_denom);

        if (eval_pdf_value > 1e-8) {
            float3 expected_weight = phase_function.eval(wi, sample.wo_ws) / eval_pdf_value;
            for (int c = 0; c < 3; ++c) {
                float weight_denom = max(max(abs(sample.weight[c]), abs(expected_weight[c])), 1e-8);
                max_weight_error = max(max_weight_error, abs(expected_weight[c] - sample.weight[c]) / weight_denom);
            }
        }

        max_direction_error = max(max_direction_error, abs(length(sample.wo_ws) - 1.0));
    }

    weight_errors.store(tid, max_weight_error);
    pdf_errors.store(tid, max_pdf_error);
    direction_errors.store(tid, max_direction_error);
}

} // namespace phase_function_tests
