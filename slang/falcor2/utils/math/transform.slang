// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Concatenates two transforms: c_from_a = c_from_b * b_from_a
public float4x4 transform_concat(float4x4 c_from_b, float4x4 b_from_a)
{
    return mul(c_from_b, b_from_a);
}

/// Concatenates two affined transforms: c_from_a = c_from_b * b_from_a
public float3x4 transform_concat(float3x4 c_from_b, float3x4 b_from_a)
{
    return float3x4(mul(float4x4(c_from_b, float4(0, 0, 0, 1)), float4x4(b_from_a, float4(0, 0, 0, 1))));
}

/// Transforms a point using an affine transform (ignores the 4th row of the matrix).
public float3 transform_point(float4x4 a_from_b, float3 point_b)
{
    return transform_point(float3x4(a_from_b), point_b);
}

/// Transforms a point using an affine transform.
public float3 transform_point(float3x4 a_from_b, float3 point_b)
{
    return mul(a_from_b, float4(point_b, 1));
}

/// Transforms a homogeneous point using a full transform.
public float4 transform_hpoint(float4x4 a_from_b, float4 hpoint_b)
{
    return mul(a_from_b, hpoint_b);
}

/// Transforms a homogeneous point using an affine transform.
public float4 transform_hpoint(float3x4 a_from_b, float4 hpoint_b)
{
    return float4(mul(a_from_b, hpoint_b), hpoint_b.w);
}

/// Transforms a vector using only the upper left 3x3 part of the matrix.
public float3 transform_vector(float4x4 a_from_b, float3 vector_b)
{
    return transform_vector(float3x3(a_from_b), vector_b);
}

/// Transforms a vector using only the upper left 3x3 part of the matrix.
public float3 transform_vector(float3x4 a_from_b, float3 vector_b)
{
    return transform_vector(float3x3(a_from_b), vector_b);
}

/// Transforms a vector.
public float3 transform_vector(float3x3 a_from_b, float3 vector_b)
{
    return mul(a_from_b, vector_b);
}


/// Transforms a normal, using the inverse transpose of the upper left 3x3 part of the matrix.
/// This is *slow* and should only be used if the transform inverse isn't available.
/// If the inverse is available, use transform_vector(transpose(float3x3(b_from_a)), normal_b) instead.
float3 transform_normal_slow(float4x4 a_from_b, float3 normal_b)
{
    float3x3 a_from_b_small = float3x3(a_from_b);
    float3x3 b_from_a_small = inverse(a_from_b_small);
    return transform_vector(transpose(b_from_a_small), normal_b);
}
