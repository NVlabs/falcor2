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
