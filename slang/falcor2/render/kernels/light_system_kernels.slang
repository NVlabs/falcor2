// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;

/// Compute the emitted power of each environment light.
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_environment_light_powers(
    uint3 tid: SV_DispatchThreadID,
    uniform LightSystem light_system,
    uniform uint light_count,
    RWStructuredBuffer<float> powers
)
{
    if (tid.x >= light_count)
        return;

    powers[tid.x] = light_system.get_environment_light(tid.x).emitted_power();
}
