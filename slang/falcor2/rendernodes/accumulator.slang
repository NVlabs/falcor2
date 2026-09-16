// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module accumulator;

import slangpy;
import falcor2.rendernodes.accumulator_common;

public struct AccumHistory4
{
    float4 sum;
    float4 residual;
};

float neumaier_correction(float sum, float value, float next)
{
    float correction;
    if (abs(sum) >= abs(value))
        correction = fma(-1.f, next, sum) + value;
    else
        correction = fma(-1.f, next, value) + sum;
    return correction;
}

public static void accumulator_update_single<T : __BuiltinFloatingPointType>(
    vector<T, 4> input,
    inout float4 sum,
    uniform uint sample_count,
    uniform bool sanitize_non_finite,
    uniform float max_sample_luminance,
    out vector<T, 4> output
)
{
    float3 input_value = prepare_accumulation_sample(float3(input.xyz), sanitize_non_finite, max_sample_luminance);
    if (sample_count == 0) {
        sum = float4(input_value, 0.f);
        output = vector<T, 4>(float4(input_value, 1.f));
        return;
    }

    sum = float4(sum.xyz + input_value, 0.f);
    float inverse_count = 1.f / float(sample_count + 1);
    output = vector<T, 4>(float4(sum.xyz * inverse_count, 1.f));
}

public static void accumulator_update_single_compensated<T : __BuiltinFloatingPointType>(
    vector<T, 4> input,
    inout AccumHistory4 history,
    uniform uint sample_count,
    uniform bool sanitize_non_finite,
    uniform float max_sample_luminance,
    out vector<T, 4> output
)
{
    float3 input_value = prepare_accumulation_sample(float3(input.xyz), sanitize_non_finite, max_sample_luminance);
    if (sample_count == 0) {
        history.sum = float4(input_value, 0.f);
        history.residual = 0.f;
        output = vector<T, 4>(float4(input_value, 1.f));
        return;
    }

    float3 next = history.sum.xyz + input_value;
    history.residual.xyz += float3(
        neumaier_correction(history.sum.x, input_value.x, next.x),
        neumaier_correction(history.sum.y, input_value.y, next.y),
        neumaier_correction(history.sum.z, input_value.z, next.z)
    );
    history.sum = float4(next, 0.f);
    history.residual.w = 0.f;
    float inverse_count = 1.f / float(sample_count + 1);
    output = vector<T, 4>(float4((next + history.residual.xyz) * inverse_count, 1.f));
}
