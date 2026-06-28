// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Computes new ray origin based on hit position to avoid self-intersections.
/// The function assumes that the hit position has been computed by barycentric
/// interpolation, and not from the ray t which is less accurate.
///
/// The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
/// Method for Avoiding Self-Intersection" by Carsten Wachter and Nikolaus Binder.
///
/// (Temporarily) We'll substitute a simple identity method for the gradient computation
/// since the asfloat(asint(x)) will break gradient propagation.
///
/// @param pos Ray hit position.
/// @param normal Face normal of hit surface (normalized). The offset will be in the positive direction.
/// @return Ray origin of the new ray.
[ForwardDerivative(offset_ray_origin_fwd), BackwardDerivative(offset_ray_origin_bwd), PreferRecompute, __NoSideEffect]
public float3 offset_ray_origin(float3 pos, float3 normal)
{
    const float ORIGIN = 1.0 / 16.0;
    const float FLOAT_SCALE = 3.0 / 65536.0;
    const float INT_SCALE = 3.0 * 256.0;

    // Per-component integer offset to bit representation of fp32 position.
    int3 offset_i = int3(normal * INT_SCALE);
    // TODO: Slang outputs a ternary to HLSL which dxc rejects with the message:
    // https://github.com/shader-slang/slang/issues/5771
    // error :  condition for short-circuiting ternary operator must be scalar, for non-scalar types
    // use 'select' float3 pos_i = asfloat(asint(pos) + select(pos < 0.0, -offset_i, offset_i));
    float3 pos_i = float3(
        asfloat(asint(pos[0]) + select(pos[0] < 0.0, -offset_i[0], offset_i[0])),
        asfloat(asint(pos[1]) + select(pos[1] < 0.0, -offset_i[1], offset_i[1])),
        asfloat(asint(pos[2]) + select(pos[2] < 0.0, -offset_i[2], offset_i[2])),
    );
    // Select per-component between small fixed offset or above variable offset depending on
    // distance to origin.
    float3 offset_f = normal * FLOAT_SCALE;
    // TODO: same here
    // https://github.com/shader-slang/slang/issues/5771
    // return select(abs(pos) < ORIGIN, pos + offset_f, pos_i);
    return float3(
        select(abs(pos[0]) < ORIGIN, pos[0] + offset_f[0], pos_i[0]),
        select(abs(pos[1]) < ORIGIN, pos[1] + offset_f[1], pos_i[1]),
        select(abs(pos[2]) < ORIGIN, pos[2] + offset_f[2], pos_i[2]),
    );
}

[Differentiable, PreferRecompute, __NoSideEffect]
DifferentialPair<float3> offset_ray_origin_fwd(DifferentialPair<float3> pos, DifferentialPair<float3> normal)
{
    return DifferentialPair<float3>(offset_ray_origin(pos.p, normal.p), pos.d);
}

[Differentiable, PreferRecompute, __NoSideEffect]
void offset_ray_origin_bwd(
    inout DifferentialPair<float3> pos,
    inout DifferentialPair<float3> normal,
    float3.Differential d
)
{
    pos = diffPair(pos.p, d);
    normal = diffPair(normal.p, float3.dzero());
}
