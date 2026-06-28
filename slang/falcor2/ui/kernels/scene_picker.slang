// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

#include "falcor2/render/scene_ray_tracing.slangh"

// ----------------------------------------------------------------------------
// Compute version
// ----------------------------------------------------------------------------

#ifndef __TARGET_CUDA__
[shader("compute")]
[numthreads(8, 8, 1)]
public void render_ids_compute(uint2 tid: SV_DispatchThreadID, uniform Camera camera, uniform RWTexture2D<uint> output)
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
#endif

// ----------------------------------------------------------------------------
// Raytracing version
// ----------------------------------------------------------------------------

// TODO: This is a workaround for CUDA not passing ray-gen entry point parameters correctly.
// https://github.com/shader-slang/slang/issues/10188
struct RenderIdsParams
{
    Camera camera;
    RWTexture2D<uint> output;
};
ParameterBlock<RenderIdsParams> g_render_ids_params;

[raypayload]
public struct RayPayload
{
    HitInfo hit : read(caller)
        : write(closesthit);

    __init() { hit = {}; }
};

public struct RayTracingSceneIntersector : ISceneIntersector
{
    public HitInfo intersect(Ray ray)
    {
        RayPayload payload = {};
        TraceRay(
            g_scene.tlas,
            0,
            0xff,
            0 /* RayContributionToHitGroupIndex */,
            0 /* MultiplierForGeometryContributionHitGroupIndex */,
            0 /* MissShaderIndex */,
            ray.to_ray_desc(),
            payload
        );
        return payload.hit;
    }

    public bool is_occluded(Ray ray)
    {
        HitInfo hit = intersect(ray);
        return hit.valid;
    }
};

SCENE_RAY_TYPE_MISS(intersect, intersect_miss, RayPayload);
void intersect_miss(inout RayPayload payload) { }

SCENE_RAY_TYPE_CLOSEST_HIT(intersect, intersect_closest_hit, RayPayload);
void intersect_closest_hit<D : IDeferredHitInfo>(inout RayPayload payload, D deferred_hit)
{
    payload.hit = deferred_hit.create();
}

[shader("raygeneration")]
public void render_ids_ray_gen()
{
    uint2 dims;
    g_render_ids_params.output.GetDimensions(dims.x, dims.y);

    uint2 pixel = DispatchRaysIndex().xy;
    if (any(pixel >= dims))
        return;

    float2 uv = (float2(pixel) + 0.5) / float2(dims);
    uv.y = 1 - uv.y;

    Ray ray = g_render_ids_params.camera.get_ray(uv);
    RayTracingSceneIntersector intersector;
    HitInfo hit = intersector.intersect(ray);

    g_render_ids_params.output[pixel] = hit.valid ? uint(hit.geometry_instance_id) : 0xffffffff;
}
