// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

RWTexture2D<float4> pattern;
uint width;
uint height;

float3 pattern_color(uint x)
{
    const float3 bars[8]
        = { float3(1, 1, 1),
            float3(1, 1, 0),
            float3(0, 1, 1),
            float3(0, 1, 0),
            float3(1, 0, 1),
            float3(1, 0, 0),
            float3(0, 0, 1),
            float3(0, 0, 0) };
    return bars[min(x * 8 / width, 7)];
}

[shader("compute")]
[numthreads(8, 8, 1)]
void make_pattern(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= width || tid.y >= height)
        return;
    pattern[tid.xy] = float4(pattern_color(tid.x), 1);
}
