// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Display-referred SDR RGB -> sRGB transfer -> BT.709 limited-range I420.
// Each thread owns eight pixels across two rows, so all uint stores are disjoint.
Texture2D<float4> source;
RWByteAddressBuffer i420;
uint width;
uint height;
uint pitch;
uint input_is_srgb;

float3 to_srgb(float3 value)
{
    value = saturate(value);
    return select(value <= 0.0031308, 12.92 * value, 1.055 * pow(value, 1.0 / 2.4) - 0.055);
}

float3 yuv(float3 rgb)
{
    float y = dot(rgb, float3(0.2126, 0.7152, 0.0722));
    return float3(16.0 + 219.0 * y, 128.0 + 224.0 * (rgb.b - y) / 1.8556, 128.0 + 224.0 * (rgb.r - y) / 1.5748);
}

// Four chroma samples per thread keep both planar stores uint-aligned.
[shader("compute")]
[numthreads(8, 8, 1)]
void convert_i420(uint3 tid: SV_DispatchThreadID)
{
    uint x = tid.x * 8;
    uint y = tid.y * 2;
    if (x >= pitch || y >= height)
        return;
    uint packed_u = 0;
    uint packed_v = 0;
    uint packed_y[4] = { 0, 0, 0, 0 };
    for (uint pair = 0; pair < 4; ++pair) {
        float2 uv = float2(0);
        for (uint row = 0; row < 2; ++row) {
            for (uint col = 0; col < 2; ++col) {
                uint dx = pair * 2 + col;
                float3 rgb = source[uint2(min(x + dx, width - 1), y + row)].rgb;
                float3 value = yuv(input_is_srgb != 0 ? saturate(rgb) : to_srgb(rgb));
                packed_y[row * 2 + dx / 4] |= uint(round(clamp(value.x, 16.0, 235.0))) << ((dx % 4) * 8);
                uv += value.yz * 0.25;
            }
        }
        packed_u |= uint(round(clamp(uv.x, 16.0, 240.0))) << (pair * 8);
        packed_v |= uint(round(clamp(uv.y, 16.0, 240.0))) << (pair * 8);
    }
    for (uint row = 0; row < 2; ++row) {
        i420.Store((y + row) * pitch + x, packed_y[row * 2]);
        i420.Store((y + row) * pitch + x + 4, packed_y[row * 2 + 1]);
    }
    uint chroma_offset = tid.y * (pitch / 2) + x / 2;
    i420.Store(height * pitch + chroma_offset, packed_u);
    i420.Store(height * pitch * 5 / 4 + chroma_offset, packed_v);
}
