// SPDX-License-Identifier: Apache-2.0

implementing rendering;

/// Sampler for environment maps in the scene.
/// Uses hierarchical importance sampling.
public struct EnvMapSampler
{
    [Differentiable]
    public float3 eval(no_diff float3 dir_ws) { return g_scene.env_map.eval(dir_ws); }

    [Differentiable, PreferRecompute, __NoSideEffect]
    public bool sample(no_diff inout UniformSampleGenerator sg, no_diff out DirectionSample sample, out float3 weight)
    {
        sample = {};
        weight = {};

        float2 inv_dim = 1.0 / g_scene.env_map.importance_map_dims.xy;
        float2 p = no_diff next_float2(sg);
        uint2 pos = 0; // Top-left texel pos of current 2x2 region.

        // Iterate over mips of 2x2...NxN resolution.
        // TODO MaxIters depends on the mip count of the importance map.
        [MaxIters(10)]
        for (int mip = g_scene.env_map.importance_map_dims.z - 1; mip >= 0; mip--) {
            // Scale position to current mip.
            pos *= 2;

            // Load the four texels at the current position.
            float w[4];
            w[0] = no_diff g_scene.env_map.importance_map.Load(int3(pos, mip));
            w[1] = no_diff g_scene.env_map.importance_map.Load(int3(pos + uint2(1, 0), mip));
            w[2] = no_diff g_scene.env_map.importance_map.Load(int3(pos + uint2(0, 1), mip));
            w[3] = no_diff g_scene.env_map.importance_map.Load(int3(pos + uint2(1, 1), mip));

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
                p.x = (p.x - d) / (1.f - d);
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
                p.y = (p.y - e) / (1.f - e);
            }

            pos += off;
        }

        // At this point, we have chosen a texel 'pos' in the range [0,dimension) for each component.
        // The 2D sample point 'p' has been warped along the way, and is in the range [0,1) representing sub-texel
        // location.

        // Compute final sample position and map to direction.
        float2 uv = (float2(pos) + p) * inv_dim; // Final sample in [0,1)^2.
        float3 dir = oct_to_ndir_equal_area_unorm(uv);

        // 1x1 mip holds integral over importance map.
        float avg_w
            = no_diff g_scene.env_map.importance_map.Load(int3(0, 0, g_scene.env_map.importance_map_dims.z - 1));

        // Compute final pdf.
        // We sample exactly according to the intensity of where the final samples lies in the octahedral map,
        // normalized to its average intensity.
        float pdf = no_diff g_scene.env_map.importance_map.Load(int3(pos, 0)) / avg_w;

        sample.emitter_type = EmitterType::env_map;
        sample.dir_ws = g_scene.env_map.local_to_world(dir);
        sample.pdf = pdf / float.FOUR_PI;
        weight = g_scene.env_map.eval(sample.dir_ws) / sample.pdf;

        return true;
    }

    float eval_pdf(DirectionSample sample)
    {
        float2 uv = ndir_to_oct_equal_area_unorm(g_scene.env_map.world_to_local(sample.dir_ws));
        float avg_w = g_scene.env_map.importance_map.Load(int3(0, 0, g_scene.env_map.importance_map_dims.z - 1));
        uint2 pos = uint2(uv * g_scene.env_map.importance_map_dims.xy);
        float pdf = g_scene.env_map.importance_map.Load(int3(pos, 0)) / avg_w;
        return pdf / float.FOUR_PI;
    }
}
