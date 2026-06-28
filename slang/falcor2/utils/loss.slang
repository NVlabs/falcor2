// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

[Differentiable]
public T l1_loss<T : __BuiltinFloatingPointType, let N : int>(vector<T, N> pred, vector<T, N> ref)
{
    vector<T, N> diff = abs(ref - pred);
    T sum = T(0);
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
        sum += diff[i];
    return sum / T(N);
}

[Differentiable]
public T l2_loss<T : __BuiltinFloatingPointType, let N : int>(vector<T, N> pred, vector<T, N> ref)
{
    vector<T, N> diff = ref - pred;
    T sum = T(0);
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
        sum += diff[i] * diff[i];
    return sum / T(N);
}

[Differentiable]
public float test_loss(float4 pred, float4 ref)
{
    return l2_loss(pred.xyz, ref.xyz);
}
