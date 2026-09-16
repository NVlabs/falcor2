// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;

/// Compute the power estimate of each component light for importance sampling.
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_light_powers(
    uint3 tid: SV_DispatchThreadID,
    uniform LightSystem light_system,
    uniform uint light_count,
    RWStructuredBuffer<float> powers
)
{
    if (tid.x >= light_count)
        return;

    powers[tid.x] = light_system.get_light(LightID(tid.x)).power_estimate();
}
