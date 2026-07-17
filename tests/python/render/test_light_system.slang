// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;

float environment_light_selection_pdf(uint index)
{
    return g_scene.light_system.environment_light_selection_distribution.eval_pdf(index);
}

float environment_direction_pdf(float3 direction)
{
    SurfaceInteraction ref = {};
    DirectionSample sample = {};
    sample.dir_ws = direction;
    sample.emitter_type = EmitterType::environment;
    LightSampler<false, true, false> light_sampler = {};
    return light_sampler.eval_environment_direction_pdf(ref, sample);
}
