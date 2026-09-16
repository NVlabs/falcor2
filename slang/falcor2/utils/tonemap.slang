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
