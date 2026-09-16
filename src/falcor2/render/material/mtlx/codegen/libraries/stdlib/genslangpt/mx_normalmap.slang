// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

void mx_normalmap_vector2(
    float3 value,
    float2 normal_scale,
    float3 normal,
    float3 tangent,
    float3 bitangent,
    out float3 result
)
{
    float3 v = select(dot(value, value) == 0.0, float3(0.0, 0.0, 1.0), value * 2.0 - 1.0);
    v.xy *= normal_scale;
    result = normalize(tangent * v.x + bitangent * v.y + normal * v.z);
}

void mx_normalmap_float(
    float3 value,
    float normal_scale,
    float3 normal,
    float3 tangent,
    float3 bitangent,
    out float3 result
)
{
    mx_normalmap_vector2(value, float2(normal_scale), normal, tangent, bitangent, result);
}
