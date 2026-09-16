// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module nan_overlay;

import slangpy;

uint2 map_pixel(uint2 pixel, uint2 source_size, uint2 target_size)
{
    return min(pixel * source_size / target_size, source_size - 1u);
}

bool has_nan_count(uint2 pixel, Texture2D<uint> nan_count, uint2 nan_count_size, uint2 output_size)
{
    uint2 nan_pixel = map_pixel(pixel, nan_count_size, output_size);
    return nan_count[nan_pixel] != 0u;
}

float3 nan_heat(uint count)
{
    float heat = saturate(log2(float(count)) / 6.0);
    if (heat < 0.5)
        return lerp(float3(1.0, 1.0, 0.0), float3(1.0, 0.0, 0.0), heat * 2.0);
    return lerp(float3(1.0, 0.0, 0.0), float3(1.0, 1.0, 1.0), heat * 2.0 - 1.0);
}

public void apply_nan_overlay(uint2 pixel, Texture2D<uint> nan_count, RWTexture2D<float4> output)
{
    uint output_width;
    uint output_height;
    output.GetDimensions(output_width, output_height);
    uint2 output_size = uint2(output_width, output_height);

    uint nan_width;
    uint nan_height;
    nan_count.GetDimensions(nan_width, nan_height);
    uint2 nan_count_size = uint2(nan_width, nan_height);
    uint2 nan_pixel = map_pixel(pixel, nan_count_size, output_size);
    uint count = nan_count[nan_pixel];
    if (count != 0u) {
        output[pixel] = float4(nan_heat(count), 1.0);
        return;
    }

    int2 min_pixel = int2(0, 0);
    int2 max_pixel = int2(output_size) - 1;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            int2 neighbor = clamp(int2(pixel) + int2(x, y), min_pixel, max_pixel);
            if (has_nan_count(uint2(neighbor), nan_count, nan_count_size, output_size)) {
                output[pixel] = float4(0.0, 0.0, 0.0, 1.0);
                return;
            }
        }
    }
}
