// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Invalid phase-function sentinel used when a medium has no phase function.
/// It evaluates to zero, has zero sampling density, and cannot produce a sample.
public struct NullPhaseFunction : IPhaseFunction
{
    public __init() { }

    public float3 eval(const float3 wi_ws, const float3 wo_ws) { return 0.0; }

    public bool sample<S : ISampleGenerator>(const float3 wi_ws, inout S sg, out PhaseFunctionSample sample)
    {
        sample = {};
        return false;
    }

    public float eval_pdf(const float3 wi_ws, const float3 wo_ws) { return 0.0; }
};
