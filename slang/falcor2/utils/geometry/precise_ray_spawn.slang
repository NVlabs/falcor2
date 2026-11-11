// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

implementing utils;

// Shared triangle/single-instance adaptation of NVIDIA's self-intersection
// avoidance code. No scene, material, traversal callback or OptiX dependency.
// https://github.com/NVIDIA/optix-toolkit/tree/master/ShaderUtil/include/OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance
// https://github.com/NVIDIA/self-intersection-avoidance
//
// Keep arithmetic order explicit. CUDA uses native directed rounding. Other
// targets conservatively round each separate operation outward; this need not
// be bit-identical to CUDA. The common coefficients use the more conservative
// HLSL sample values, not the Toolkit's tighter CUDA-specific triangle bound.
// Assumptions: finite, nondegenerate triangles and matching affine matrices for
// the traversal instance. This does not bound arbitrary matrix inversion errors,
// motion transforms or non-triangle intersection algorithms.

namespace precise_ray_spawn_detail {
// Step toward an outer representable value. Skip subnormals so a backend
// flushing them to zero cannot undo outward rounding. Zero operands are
// handled by callers to avoid perturbing already-exact zero operations.
public float next_up(float value)
{
    if (isnan(value) || value == asfloat(0x7f800000u))
        return value;
    if (abs(value) < asfloat(0x00800000u))
        return asfloat(0x00800000u);
    uint bits = asuint(value);
    return asfloat(value > 0.0 ? bits + 1u : bits - 1u);
}

public float next_down(float value)
{
    return -next_up(-value);
}

public float add_rn(float a, float b)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__fadd_rn($0, $1)";
    default:
        precise float result = a + b;
        return result;
    }
}

public float mul_rn(float a, float b)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__fmul_rn($0, $1)";
    default:
        precise float result = a * b;
        return result;
    }
}

public float mad_rn(float a, float b, float c)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__fmaf_rn($0, $1, $2)";
    default:
        precise float result = fma(a, b, c);
        return result;
    }
}

public float add_up(float a, float b)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__fadd_ru($0, $1)";
    default:
        if (a == 0.0)
            return b;
        if (b == 0.0)
            return a;
        return next_up(add_rn(a, b));
    }
}

public float mul_up(float a, float b)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__fmul_ru($0, $1)";
    default:
        if (a == 0.0 || b == 0.0)
            return 0.0;
        return next_up(mul_rn(a, b));
    }
}

public float mad_up(float a, float b, float c)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__fmaf_ru($0, $1, $2)";
    default:
        // Two outward-rounded operations also bound a non-fused HLSL mad.
        return add_up(mul_up(a, b), c);
    }
}

public float mad_down(float a, float b, float c)
{
    return -mad_up(-a, b, -c);
}

float dot_rn(float3 a, float3 b)
{
    return mad_rn(a.x, b.x, mad_rn(a.y, b.y, mul_rn(a.z, b.z)));
}

float dot_abs_up(float3 a, float3 b)
{
    a = abs(a);
    b = abs(b);
    return mad_up(a.x, b.x, mad_up(a.y, b.y, mul_up(a.z, b.z)));
}

float difference_of_products(float a, float b, float c, float d)
{
    float cd = mul_rn(c, d);
    float error = mad_rn(-c, d, cd);
    return add_rn(mad_rn(a, b, -cd), error);
}

float3 cross_rn(float3 a, float3 b)
{
    return float3(
        difference_of_products(a.y, b.z, a.z, b.y),
        difference_of_products(a.z, b.x, a.x, b.z),
        difference_of_products(a.x, b.y, a.y, b.x)
    );
}

float3 scale_rn(float3 value, float scale)
{
    return float3(mul_rn(value.x, scale), mul_rn(value.y, scale), mul_rn(value.z, scale));
}
}

/// A reconstructed triangle point and the offset along its unit geometric normal.
public struct PreciseRaySpawn
{
    public float3 position = {};
    public float3 normal = {};
    /// Zero means that this input did not produce a finite, positive bound.
    public float offset = 0.0;
}

/// Compute a conservative spawn offset for a triangle in a single affine instance.
public PreciseRaySpawn compute_precise_ray_spawn(
    float3 v0,
    float3 v1,
    float3 v2,
    float2 barycentrics,
    float4x4 object_to_world,
    float4x4 world_to_object
)
{
    using namespace precise_ray_spawn_detail;
    PreciseRaySpawn result = {};
    float3 edge1, edge2, object_position;
    // Unroll in Slang to avoid dynamically indexed local arrays on D3D12.
    [ForceUnroll]
    for (uint i = 0; i < 3; ++i) {
        edge1[i] = add_rn(v1[i], -v0[i]);
        edge2[i] = add_rn(v2[i], -v0[i]);
        object_position[i] = add_rn(v0[i], mad_rn(barycentrics.x, edge1[i], mul_rn(barycentrics.y, edge2[i])));
    }

    float3 object_normal = cross_rn(edge1, edge2);
    float normal_squared = dot_rn(object_normal, object_normal);
    if (!(normal_squared > 0.0) || !isfinite(normal_squared))
        return result;
    object_normal = scale_rn(object_normal, rsqrt(normal_squared));

    static const float C0 = 5.9604648328104529e-8;
    static const float C1 = 1.788139769587360206060111522674560546875e-7;
    static const float C2 = 1.19209317972490680404007434844970703125e-7;
    float3 extent;
    [ForceUnroll]
    for (uint i = 0; i < 3; ++i) {
        // Bound the magnitude of the difference in either rounding direction.
        float difference = max(abs(add_up(edge1[i], -edge2[i])), abs(-add_up(-edge1[i], edge2[i])));
        extent[i] = add_up(add_up(abs(edge1[i]), abs(edge2[i])), difference);
    }
    float intersection_error = mul_up(C1, max(extent.x, max(extent.y, extent.z)));
    float3 object_error, world_error, world_normal;
    [ForceUnroll]
    for (uint i = 0; i < 3; ++i) {
        object_error[i] = mad_up(C0, abs(v0[i]), intersection_error);
        result.position[i] = add_rn(object_to_world[i][3], dot_rn(object_to_world[i].xyz, object_position));
        world_error[i]
            = mad_up(C1, dot_abs_up(object_to_world[i].xyz, object_position), mul_up(C2, abs(object_to_world[i][3])));
        world_normal[i]
            = dot_rn(float3(world_to_object[0][i], world_to_object[1][i], world_to_object[2][i]), object_normal);
    }
    float object_offset = dot_abs_up(object_error, object_normal);
    [ForceUnroll]
    for (uint i = 0; i < 3; ++i)
        object_error[i]
            = mul_up(C2, add_up(dot_abs_up(world_to_object[i].xyz, result.position), abs(world_to_object[i][3])));
    object_offset = add_up(object_offset, dot_abs_up(object_error, object_normal));

    float world_normal_squared = dot_rn(world_normal, world_normal);
    if (!(world_normal_squared > 0.0) || !isfinite(world_normal_squared))
        return result;
    float normal_scale = rsqrt(world_normal_squared);
    result.normal = scale_rn(world_normal, normal_scale);
    float offset = mad_up(object_offset, normal_scale, dot_abs_up(world_error, result.normal));
    if (isfinite(offset) && all(isfinite(result.position)))
        result.offset = offset;
    return result;
}

/// Offset toward the oriented geometric normal, rounding away from the surface.
public float3 offset_ray_origin_precise(float3 position, float3 oriented_normal, float offset)
{
    float3 result;
    [ForceUnroll]
    for (uint i = 0; i < 3; ++i) {
        if (oriented_normal[i] == 0.0)
            result[i] = position[i];
        else if (oriented_normal[i] > 0.0)
            result[i] = precise_ray_spawn_detail::mad_up(offset, oriented_normal[i], position[i]);
        else
            result[i] = precise_ray_spawn_detail::mad_down(offset, oriented_normal[i], position[i]);
    }
    return result;
}
