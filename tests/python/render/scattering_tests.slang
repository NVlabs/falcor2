// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
__exported import falcor2.render;
__exported import falcor2.utils;

namespace scattering_tests {

/// Test adapters preserve rejection and express all directions in one orthonormal frame.
public interface IScatteringTestTarget
{
    bool sample(BSDFSampleInput input, out BSDFSample result);

    BSDFEval eval_with_pdf(float3 wo, BSDFComponent component = BSDFComponent::full_response);
}

public struct ContractCheck
{
    // 0: invalid, 1: continuous, 2: delta. Rejection is handled before validation.
    public uint category;
    public float weight_error;
    public float pdf_error;
};

/// Errors are fractions of the allowed absolute-plus-relative tolerance.
public ContractCheck check_sample(BSDFSample sample, BSDFEval evaluated, float weight_rtol, float pdf_rtol, float atol)
{
    ContractCheck result = {};
    if (!isfinite(sample.pdf) || sample.pdf < 0.0 || !all(isfinite(sample.weight)) || any(sample.weight < 0.0)
        || !all(isfinite(sample.wo)) || abs(dot(sample.wo, sample.wo) - 1.0) > 2.0e-3)
        return result;

    if (sample.has_flag(BSDFFlags::delta)) {
        result.category = sample.pdf == 0.0 ? 2 : 0;
        return result;
    }

    if (sample.pdf <= 0.0 || !isfinite(evaluated.pdf) || evaluated.pdf <= 0.0 || !all(isfinite(evaluated.value))
        || any(evaluated.value < 0.0))
        return result;

    result.category = 1;
    result.pdf_error = abs(sample.pdf - evaluated.pdf) / (atol + pdf_rtol * max(sample.pdf, evaluated.pdf));
    const float3 reconstructed = sample.weight * sample.pdf;
    if (!all(isfinite(reconstructed))) {
        result.category = 0;
        return result;
    }
    const float3 error
        = abs(reconstructed - evaluated.value) / (atol + weight_rtol * max(abs(reconstructed), abs(evaluated.value)));
    result.weight_error = max(error.x, max(error.y, error.z));
    return result;
}

/// Counts include every attempt: continuous, rejected, delta, invalid.
public void test_sample_contract<T : IScatteringTestTarget>(
    uint tid,
    uint sample_count,
    uint samples_per_thread,
    uint seed,
    T target,
    float weight_rtol,
    float pdf_rtol,
    float atol,
    WTensor<float, 1> weight_errors,
    WTensor<float, 1> pdf_errors,
    AtomicTensor<uint, 1> counters
)
{
    const uint2 v = block_cipher_tea(seed, tid);
    var sg = UniformSampleGenerator(v.x);
    float max_weight_error = 0.0;
    float max_pdf_error = 0.0;
    for (uint i = 0; i < samples_per_thread && tid * samples_per_thread + i < sample_count; ++i) {
        BSDFSample sample = {};
        if (!target.sample(make_bsdf_sample_input(sg), sample)) {
            counters.add(1, 1);
            continue;
        }
        const BSDFEval evaluated = sample.has_flag(BSDFFlags::delta) ? BSDFEval() : target.eval_with_pdf(sample.wo);
        ContractCheck checked = check_sample(sample, evaluated, weight_rtol, pdf_rtol, atol);
        if (checked.category == 1 && sample.component != BSDFComponent::full_response) {
            const BSDFEval selected = target.eval_with_pdf(sample.wo, sample.component);
            BSDFSample joint_sample = sample;
            joint_sample.pdf = selected.pdf;
            const ContractCheck component_checked = check_sample(joint_sample, selected, weight_rtol, pdf_rtol, atol);
            checked.category = component_checked.category;
            checked.weight_error = component_checked.weight_error;
        }
        counters.add(checked.category == 1 ? 0 : checked.category == 2 ? 2 : 3, 1);
        max_weight_error = max(max_weight_error, checked.weight_error);
        max_pdf_error = max(max_pdf_error, checked.pdf_error);
    }
    weight_errors.store(tid, max_weight_error);
    pdf_errors.store(tid, max_pdf_error);
}

/// Independently sum a material's fixed component slots at arbitrary directions.
public float4 component_partition<T : IScatteringTestTarget>(T target, float3 wo, uint component_count)
{
    BSDFEval sum = {};
    for (uint i = 0; i < component_count; ++i) {
        const BSDFEval part = target.eval_with_pdf(wo, BSDFComponent(i));
        sum.value += part.value;
        sum.pdf += part.pdf;
    }
    const BSDFEval full = target.eval_with_pdf(wo);
    return float4(sum.value - full.value, sum.pdf - full.pdf);
}

public float4 component_value<T : IScatteringTestTarget>(T target, float3 wo, uint component)
{
    const BSDFEval evaluated = target.eval_with_pdf(wo, BSDFComponent(component));
    return float4(evaluated.value, evaluated.pdf);
}

/// Delta events are excluded from the continuous integral; rejected attempts contribute zero.
public float3 sample_continuous_weight<T : IScatteringTestTarget>(T target, uint3 seed)
{
    var sg = UniformSampleGenerator(seed.xy, seed.z);
    BSDFSample result = {};
    if (!target.sample(make_bsdf_sample_input(sg), result) || result.has_flag(BSDFFlags::delta))
        return float3(0.0);
    return result.weight;
}

public float3 eval_value<T : IScatteringTestTarget>(T target, float3 wo)
{
    return target.eval_with_pdf(wo).value;
}

public float4 sample_event<T : IScatteringTestTarget>(T target, BSDFSampleInput input)
{
    BSDFSample result = {};
    if (!target.sample(input, result))
        return float4(0.0);
    return float4(result.wo, float(uint(result.flags)));
}

public float4 sample_event_weight<T : IScatteringTestTarget>(T target, BSDFSampleInput input)
{
    BSDFSample result = {};
    if (!target.sample(input, result))
        return float4(0.0, 0.0, 0.0, -1.0);
    return float4(result.weight, result.pdf);
}

public uint sample_component<T : IScatteringTestTarget>(T target, BSDFSampleInput input)
{
    BSDFSample result = {};
    if (!target.sample(input, result))
        return uint(BSDFComponent::full_response);
    return uint(result.component);
}

} // namespace scattering_tests
