// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct NullPhaseFunction : IPhaseFunction
{
    public __init() { }

    public float3 eval(const float3 wi_ws, const float3 wo_ws) { return 0.f; }

    public bool
    sample<S : ISampleGenerator>(const float3 wi_ws, out float3 wo_ws, out float pdf, out float3 weight, inout S sg)
    {
        wo_ws = -wi_ws;
        pdf = 1.f;
        weight = 1.f;
        return true;
    }

    public float eval_pdf(const float3 wi_ws, const float3 wo_ws) { return 0.f; }
};
