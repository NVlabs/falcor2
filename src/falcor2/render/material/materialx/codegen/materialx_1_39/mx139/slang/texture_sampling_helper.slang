// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mx139_snippets;

// OSL-derived profiles use the TextureHandle sampler for MaterialX address/filter modes.
// The analytic sphere visualizer's LOD sampler forces OSL texture(..., "width", 0.0) base-LOD sampling.
float4 sample_texture<TLodSampler : ILodSampler>(
    TextureHandle handle,
    float2 uv,
    TLodSampler lod_sampler,
    float4 default_value = {}
)
{
    return handle.sample(lod_sampler, uv, default_value);
}
