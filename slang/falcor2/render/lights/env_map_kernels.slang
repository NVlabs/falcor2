// SPDX-License-Identifier: Apache-2.0

import falcor2.utils;
import falcor2.render;

// TODO: build_importance_map should take a texture handle for the env_map but we have
// bindless issues on linux/vulkan.
[shader("compute")]
[numthreads(16, 16, 1)]
public void build_importance_map(
    uint3 tid: SV_DispatchThreadID,
    Texture2D<float4> env_map,
    SamplerState env_map_sampler,
    RWTexture2D<float> importance_map,
    uniform uint sample_count
)
{
    uint2 dims;
    importance_map.GetDimensions(dims.x, dims.y);
    if (any(tid.xy >= dims))
        return;

    float L = 0.0;
    for (uint x = 0; x < sample_count; ++x) {
        for (uint y = 0; y < sample_count; ++y) {
            float2 p = float2(tid.x * sample_count + x, tid.y * sample_count + y) / (dims * sample_count);
            float3 dir = oct_to_ndir_equal_area_unorm(p);
            float2 uv = world_to_latlong_map(dir);
            float3 c = env_map.SampleLevel(env_map_sampler, uv, 0).rgb;
            L += luminance(c);
        }
    }
    L /= (sample_count * sample_count);
    importance_map[tid.xy] = L;
}
