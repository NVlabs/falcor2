// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module accumulator_double;

import slangpy;
import falcor2.rendernodes.accumulator_common;

public static void accumulator_update_double<T : __BuiltinFloatingPointType>(
    vector<T, 4> input,
    inout double4 sum,
    uniform uint sample_count,
    uniform bool sanitize_non_finite,
    uniform float max_sample_luminance,
    out vector<T, 4> output
)
{
    float3 input_value = prepare_accumulation_sample(float3(input.xyz), sanitize_non_finite, max_sample_luminance);

    if (sample_count == 0) {
        sum = double4(input_value, 0.0);
        output = vector<T, 4>(float4(input_value, 1.f));
        return;
    }

    sum.xyz += double3(input_value);
    double inverse_count = 1.0 / double(sample_count + 1);
    float3 average = float3(sum.xyz * inverse_count);
    output = vector<T, 4>(float4(average, 1.f));
}
