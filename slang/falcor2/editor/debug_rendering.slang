// SPDX-License-Identifier: Apache-2.0

implementing editor;

import falcor2.render;
import falcor2.utils;

enum class RenderMode : uint {
    path_traced,
    shaded,
    flat_shaded,
    geometry_type,
    geometry_instance_id,
    primitive_index,
    position_ws,
    normal_ws,
    front_facing,
    texcoord,
    shading_normal,
    shading_tangent,
    shading_bitangent,
};

float3 colorize_shaded(float3 n, float3 light_dir = normalize(float3(1, 1, 1)))
{
    return float3(max(0.0, dot(n, light_dir)));
}

float3 colorize_uint(uint i)
{
    uint h = jenkins_hash(i);
    return (uint3(h, h >> 8, h >> 16) & 0xff) / 255.0;
}

float3 colorize_texcoord(float2 tc)
{
    return float3(frac(tc), 0.0);
}

float3 colorize_normal(float3 n)
{
    return saturate(n * 0.5 + 0.5);
}

float3 colorize_flag(bool flag)
{
    return select(flag, float3(0, 1, 0), float3(1, 0, 0));
}

float3 colorize_hit(HitInfo hit, SurfaceInteraction si, RenderMode mode)
{
    switch (mode) {
    case RenderMode::shaded:
        return colorize_shaded(si.shading_frame_ws.normal);
    case RenderMode::flat_shaded:
        return colorize_shaded(si.normal_ws);
    case RenderMode::geometry_type:
        return colorize_uint(hit.geometry_type);
    case RenderMode::geometry_instance_id:
        return colorize_uint(hit.geometry_instance_id);
    case RenderMode::primitive_index:
        return colorize_uint(hit.primitive_index);
    case RenderMode::position_ws:
        return frac(si.position_ws);
    case RenderMode::normal_ws:
        return colorize_normal(si.normal_ws);
    case RenderMode::front_facing:
        return colorize_flag(si.front_facing);
    case RenderMode::texcoord:
        return colorize_texcoord(si.uv);
    case RenderMode::shading_normal:
        return colorize_normal(si.shading_frame_ws.normal);
    case RenderMode::shading_tangent:
        return colorize_normal(si.shading_frame_ws.tangent);
    case RenderMode::shading_bitangent:
        return colorize_normal(si.shading_frame_ws.bitangent);
    default:
        return float3(0);
    }
}

#ifndef __TARGET_CUDA__
[shader("compute")]
[numthreads(8, 8, 1)]
public void
render_geometry_instance_ids(uint2 tid: SV_DispatchThreadID, uniform Camera camera, uniform RWTexture2D<uint> output)
{
    uint2 dims;
    output.GetDimensions(dims.x, dims.y);
    if (any(tid >= dims))
        return;

    float2 uv = (float2(tid) + 0.5) / float2(dims);
    uv.y = 1 - uv.y;

    Ray ray = camera.get_ray(uv);
    RayQuerySceneIntersector intersector;
    HitInfo hit = intersector.intersect(ray);

    output[tid] = hit.valid ? uint(hit.geometry_instance_id) : 0xffffffff;
}

[shader("compute")]
[numthreads(8, 8, 1)]
public void render_debug(
    uint2 tid: SV_DispatchThreadID,
    uniform Camera camera,
    uniform uint render_mode,
    uniform RWTexture2D<float4> color_output,
    uniform RWTexture2D<uint> geometry_instance_id_output
)
{
    uint2 dims;
    color_output.GetDimensions(dims.x, dims.y);
    if (any(tid >= dims))
        return;

    float2 uv = (float2(tid) + 0.5) / float2(dims);
    uv.y = 1 - uv.y;

    Ray ray = camera.get_ray(uv);
    RayQuerySceneIntersector intersector;
    HitInfo hit = intersector.intersect(ray);

    float3 color = 0.0;
    if (hit.valid) {
        SurfaceInteraction si = g_scene.make_surface_interaction(hit, -ray.dir);
        color = colorize_hit(hit, si, RenderMode(render_mode));
    }

    color_output[tid] = float4(color, 1.0);
    geometry_instance_id_output[tid] = hit.valid ? uint(hit.geometry_instance_id) : 0xffffffff;
}

[shader("compute")]
[numthreads(8, 8, 1)]
public void render_debug_no_ids(
    uint2 tid: SV_DispatchThreadID,
    uniform Camera camera,
    uniform uint render_mode,
    uniform RWTexture2D<float4> color_output
)
{
    uint2 dims;
    color_output.GetDimensions(dims.x, dims.y);
    if (any(tid >= dims))
        return;

    float2 uv = (float2(tid) + 0.5) / float2(dims);
    uv.y = 1 - uv.y;

    Ray ray = camera.get_ray(uv);
    RayQuerySceneIntersector intersector;
    HitInfo hit = intersector.intersect(ray);

    float3 color = 0.0;
    if (hit.valid) {
        SurfaceInteraction si = g_scene.make_surface_interaction(hit, -ray.dir);
        color = colorize_hit(hit, si, RenderMode(render_mode));
    }

    color_output[tid] = float4(color, 1.0);
}
#endif
