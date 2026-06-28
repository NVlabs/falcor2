// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing scene;

/// Environment map.
public struct EnvMap
{
    TextureID texture_id;

    public Texture2D<float> importance_map;
    public uint3 importance_map_dims;

    /// Local-space to world-space transform.
    float4x4 ws_from_ls;
    /// World-space to local-space transform.
    float4x4 ls_from_ws;

    float3 scaling_factor;

    /// Evaluates the radiance at a given uv coordinate.
    [Differentiable, PreferRecompute, __NoSideEffect]
    public float3 eval(no_diff float2 uv) { return g_scene.sample_texture_linear(texture_id, uv).xyz * scaling_factor; }

    /// Evaluates the radiance coming from a world-space direction.
    [Differentiable, PreferRecompute, __NoSideEffect]
    public float3 eval(no_diff float3 dir_ws) { return eval(world_to_uv(dir_ws)); }

    /// Transform direction in local-space to uv coordinates.
    public float2 local_to_uv(float3 dir_ls) { return world_to_latlong_map(dir_ls); }

    /// Transform uv coordinates to direction in local-space.
    public float3 uv_to_local(float2 uv) { return latlong_map_to_world(uv); }

    /// Transform direction in world-space to uv coordinates.
    public float2 world_to_uv(float3 dir_ws) { return local_to_uv(world_to_local(dir_ws)); }

    /// Transform uv coordinates to direction in world-space.
    public float3 uv_to_world(float2 uv) { return local_to_world(uv_to_local(uv)); }

    /// Transform direction in local-space to world-space.
    public float3 local_to_world(float3 dir_ls) { return mul((float3x3)ws_from_ls, dir_ls); }

    /// Transform direction in world-space to local-space.
    public float3 world_to_local(float3 dir_ws) { return mul((float3x3)ls_from_ws, dir_ws); }
};

float3 fetch_texture(ITensor<float, 3> texture, int2 texel)
{
    texel.x = texel.x % texture.shape[1];
    texel.y = texel.y % texture.shape[0];

    float3 val;
    val.x = texture[texel.y, texel.x, 0];
    val.y = texture[texel.y, texel.x, 1];
    val.z = texture[texel.y, texel.x, 2];

    return val;
}

float3 sample_texture_linear(ITensor<float, 3> texture, no_diff float2 uv)
{
    float2 shape = float2(texture.shape[1], texture.shape[0]);
    int2 texel = int2(uv * shape);
    float2 frac = frac(uv * shape);

    float3 c00 = fetch_texture(texture, texel);
    float3 c10 = fetch_texture(texture, texel + int2(1, 0));
    float3 c01 = fetch_texture(texture, texel + int2(0, 1));
    float3 c11 = fetch_texture(texture, texel + int2(1, 1));

    return lerp(lerp(c00, c10, frac.x), lerp(c01, c11, frac.x), frac.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
public void build_importance_map(
    uint3 tid: SV_DispatchThreadID,
    uniform Tensor<float, 3> texture,
    uniform uint sample_count,
    RWTexture2D<float> importance_map
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
            float3 c = sample_texture_linear(texture, uv);
            L += luminance(c);
        }
    }
    L /= (sample_count * sample_count);
    importance_map[tid.xy] = L;
}
