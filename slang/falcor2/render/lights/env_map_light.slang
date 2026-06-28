// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;
import sgl.device.print;

public struct EnvMapLight : ILight
{
    public LightHeader header;
    public float3x3 world_from_object;
    public float3 scale;
    public TextureHandle<float4, uint(TextureHandleFlags::none)> env_map_texture_handle;
    public DescriptorHandleWAR<Texture2D<float>> importance_map_texture;

    public float3 eval_direction<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        float3 dir_os = normalize(mul(transpose(world_from_object), sample.dir_ws));
        float2 uv = world_to_latlong_map(dir_os);
        return env_map_texture_handle.sample(ExplicitLodSampler(), uv).rgb * scale;
    }

    public bool sample_direction<Interaction : IInteraction, SampleGenerator : ISampleGenerator>(
        const Interaction ref,
        inout SampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    )
    {
        sample = {};
        weight = {};

        float2 uv = {};
        float pdf = {};
        if (!importance_map_sample(importance_map_texture.unwrap(), next_float2(sg), uv, pdf))
            return false;

        float3 dir_os = oct_to_ndir_equal_area_unorm(uv);
        sample.dir_ws = normalize(mul(world_from_object, dir_os));
        sample.distance = float.MAX;
        sample.pdf = pdf * float.INV_FOUR_PI;
        sample.emitter_type = EmitterType::environment;
        weight = env_map_texture_handle.sample(ExplicitLodSampler(), world_to_latlong_map(dir_os)).rgb * scale
            / sample.pdf;
        return true;
    }

    public float eval_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        float3 dir_os = normalize(mul(transpose(world_from_object), sample.dir_ws));
        float2 uv = ndir_to_oct_equal_area_unorm(dir_os);
        return importance_map_eval_pdf(importance_map_texture.unwrap(), uv) * float.INV_FOUR_PI;
    }
};

public bool importance_map_sample(Texture2D<float> importance_map, float2 p, out float2 uv, out float pdf)
{
    uint3 dims;
    importance_map.GetDimensions(0, dims.x, dims.y, dims.z);
#ifdef __TARGET_CUDA__
    // TODO(scene) GetDimensions doesn't report mip level count on CUDA
    // https://github.com/shader-slang/slang/issues/9661
    dims.z = firstbithigh(dims.x) + 1;
#endif
    float2 inv_size = 1.0 / dims.xy;

    uint2 pos = 0; // Top-left texel pos of current 2x2 region.

    // Iterate over mips of 2x2...NxN resolution.
    // TODO MaxIters depends on the mip count of the importance map.
    [MaxIters(10)]
    for (int mip = dims.z - 1; mip >= 0; --mip) {
        // Scale position to current mip.
        pos *= 2;

        // Load the four texels at the current position.
        float w[4];
        w[0] = importance_map.Load(int3(pos, mip));
        w[1] = importance_map.Load(int3(pos + uint2(1, 0), mip));
        w[2] = importance_map.Load(int3(pos + uint2(0, 1), mip));
        w[3] = importance_map.Load(int3(pos + uint2(1, 1), mip));

        float q[2];
        q[0] = w[0] + w[2];
        q[1] = w[1] + w[3];

        uint2 off;

        // Horizontal warp.
        // TODO: Do we need to guard against div-by-zero. We should ensure we never
        // go down a path that has p=0.
        float d = q[0] / (q[0] + q[1]);

        if (p.x < d) {
            // left
            off.x = 0;
            p.x = p.x / d;
        } else {
            // right
            off.x = 1;
            p.x = (p.x - d) / (1.0 - d);
        }

        // Vertical warp.
        // Avoid stack allocation by not using dynamic indexing.
        // float e = w[off.x] / q[off.x];
        float e = off.x == 0 ? (w[0] / q[0]) : (w[1] / q[1]);

        if (p.y < e) {
            // bottom
            off.y = 0;
            p.y = p.y / e;
        } else {
            // top
            off.y = 1;
            p.y = (p.y - e) / (1.0 - e);
        }

        pos += off;
    }

    // At this point, we have chosen a texel 'pos' in the range [0,dimension) for each component.
    // The 2D sample point 'p' has been warped along the way, and is in the range [0,1) representing sub-texel
    // location.

    // Compute final sample position in [0,1)^2.
    uv = (float2(pos) + p) * inv_size;

    // 1x1 mip holds integral over importance map.
    float avg = importance_map.Load(int3(0, 0, dims.z - 1));

    // Compute PDF.
    pdf = importance_map.Load(int3(pos, 0)) / avg;

    return true;
}

float importance_map_eval_pdf(Texture2D<float> importance_map, float2 uv)
{
    uint3 dims;
    importance_map.GetDimensions(0, dims.x, dims.y, dims.z);
#ifdef __TARGET_CUDA__
    // TODO(scene) GetDimensions doesn't report mip level count on CUDA
    // https://github.com/shader-slang/slang/issues/9661
    dims.z = firstbithigh(dims.x) + 1;
#endif
    uint2 pos = uint2(uv * dims.xy);

    // 1x1 mip holds integral over importance map.
    float avg = importance_map.Load(int3(0, 0, dims.z - 1));

    // Compute PDF.
    return importance_map.Load(int3(pos, 0)) / avg;
}
