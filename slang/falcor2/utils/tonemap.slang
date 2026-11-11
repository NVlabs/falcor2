// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

public float3 tonemap_aces_film(float3 color)
{
    // Cancel out the pre-exposure mentioned in
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

public float4 tonemap_aces_film(float4 color)
{
    return float4(tonemap_aces_film(color.xyz), color.w);
}

public half4 tonemap_aces_film(half4 color)
{
    return half4(half3(tonemap_aces_film(float3(color.xyz))), color.w);
}

/// Approximate Blender AgX for scene-linear Rec.709 input and linear Rec.709 output.
/// Adapted from Google Filament's AgxToneMapper (Copyright (C) 2021 The Android
/// Open Source Project, Apache-2.0):
/// https://github.com/google/filament/blob/main/filament/src/ToneMapper.cpp
public float3 tonemap_agx(float3 color)
{
    static const float3x3 rec709_to_rec2020 = {
        0.627403895934699,
        0.329283038377884,
        0.043313065687417,
        0.069097289358232,
        0.919540395075459,
        0.011362315566309,
        0.016391438875150,
        0.088013307877226,
        0.895595253247624,
    };
    static const float3x3 rec2020_to_rec709 = {
        1.660491002108435,
        -0.587641138788550,
        -0.072849863319885,
        -0.124550474521591,
        1.132899897125960,
        -0.008349422604369,
        -0.018150763354905,
        -0.100578898008007,
        1.118729661362913,
    };
    static const float3x3 agx_inset = {
        0.856627153315983,
        0.095121240538159,
        0.048251606145858,
        0.137318972929847,
        0.761241990602591,
        0.101439036467562,
        0.111898212999950,
        0.076799418603190,
        0.811302368396859,
    };
    static const float3x3 agx_outset = {
        1.127100581814437,
        -0.110606643096603,
        -0.016493938717835,
        -0.141329763498438,
        1.157823702216272,
        -0.016493938717834,
        -0.141329763498438,
        -0.110606643096603,
        1.251936406595040,
    };

    float3 v = mul(rec709_to_rec2020, max(color, 0.0));
    v = mul(agx_inset, v);
    v = saturate((log2(max(v, 1.0e-10)) + 12.47393) / 16.5);

    float3 v2 = v * v;
    float3 v4 = v2 * v2;
    float3 v6 = v4 * v2;
    v = -17.86 * v6 * v + 78.01 * v6 - 126.7 * v4 * v + 92.06 * v4 - 28.72 * v2 * v + 4.361 * v2 - 0.1718 * v
        + 0.002857;

    v = mul(agx_outset, v);
    v = pow(max(v, 0.0), float3(2.2));
    return saturate(mul(rec2020_to_rec709, v));
}

public float4 tonemap_agx(float4 color)
{
    return float4(tonemap_agx(color.xyz), color.w);
}

public half4 tonemap_agx(half4 color)
{
    return half4(half3(tonemap_agx(float3(color.xyz))), color.w);
}

public float3 tonemap_reinhard(float3 color)
{
    color = max(color, 0.f);
    return color / (1.f + color);
}

public float4 tonemap_reinhard(float4 color)
{
    return float4(tonemap_reinhard(color.xyz), color.w);
}

public half4 tonemap_reinhard(half4 color)
{
    return half4(half3(tonemap_reinhard(float3(color.xyz))), color.w);
}

public float3 tonemap_clamp(float3 color)
{
    return saturate(color);
}

public float4 tonemap_clamp(float4 color)
{
    return float4(tonemap_clamp(color.xyz), color.w);
}

public half4 tonemap_clamp(half4 color)
{
    return half4(half3(tonemap_clamp(float3(color.xyz))), color.w);
}
