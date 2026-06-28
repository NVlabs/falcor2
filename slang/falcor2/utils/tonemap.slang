// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

public float3 tonemap_aces_film(float3 x)
{
    // Cancel out the pre-exposure mentioned in
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    x *= 0.6;
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

public float4 tonemap_aces_film(float4 x)
{
    return float4(tonemap_aces_film(x.xyz), x.w);
}

// Work around for problem resolving overloads for float tensors
public float3 tonemap_aces_film_f3(float3 x)
{
    return tonemap_aces_film(x);
}
public float4 tonemap_aces_film_f4(float4 x)
{
    return tonemap_aces_film(x);
}
