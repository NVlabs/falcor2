// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mtlx_snippets;

// OSL-derived profiles use the TextureHandle sampler for MaterialX address/filter modes.
// Explicit LOD 0 samplers request base-LOD sampling, equivalent to OSL texture(..., "width", 0.0).
float4 sample_texture<TLodSampler : ILodSampler>(
    TextureHandle handle,
    float2 uv,
    TLodSampler lod_sampler,
    float4 default_value = {}
)
{
    return handle.sample(lod_sampler, uv, default_value);
}
