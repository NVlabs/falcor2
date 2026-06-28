// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

public static void accumulator_reset(out float4 accumulator)
{
    accumulator = 0;
}

public static void accumulator_output(float4 accumulator, out float4 output)
{
    output = float4(accumulator.xyz / accumulator.w, 1.0);
}

public static void accumulator_update(float4 input, inout float4 accumulator)
{
    accumulator += float4(input.xyz, 1.0);
}

public static void accumulator_update_and_output(float4 input, inout float4 accumulator, out float4 output)
{
    float4 a = accumulator;
    a += float4(input.xyz, 1.0);
    output = float4(a.xyz / a.w, 1.0);
    accumulator = a;
}

public static void accumulator_update_and_output_inplace(inout float4 input, inout float4 accumulator)
{
    float4 a = accumulator;
    a += float4(input.xyz, 1.0);
    accumulator = a;
    input = float4(a.xyz / a.w, 1.0);
}
