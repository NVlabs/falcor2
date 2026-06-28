// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
__exported import falcor2.render;

float4 test_2d(TextureHandle handle, float2 uv, float4 default_value)
{
    ExplicitLodSampler sampler;
    return handle.sample(sampler, uv, default_value);
}

float4 test_3d(TextureHandle handle, float3 uvw, float4 default_value)
{
    ExplicitLodSampler sampler;
    return handle.sample(sampler, uvw, default_value);
}

uint test_handle_flag_conversion(TextureHandle<float4, uint(TextureHandleFlags::none)> handle)
{
    TextureHandle<float4, uint(TextureHandleFlags::udim)> converted_handle = { handle };
    return converted_handle.data;
}
