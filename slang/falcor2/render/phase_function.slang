// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;
public interface IPhaseFunction
{
    /// Evaluates the phase function, in world space.
    public float3 eval(const float3 wi_ws, const float3 wo_ws);

    /// Samples the phase function, in world space.
    public bool
    sample<S : ISampleGenerator>(const float3 wi_ws, out float3 wo_ws, out float pdf, out float3 weight, inout S sg);

    /// Evaluates the phase function pdf for sampling outgoing direction wo.
    public float eval_pdf(const float3 wi_ws, const float3 wo_ws);
}
