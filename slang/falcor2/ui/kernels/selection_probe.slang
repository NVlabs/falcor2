// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

#include "falcor2/render/scene_ray_tracing.slangh"

// Helper to check if a geometry instance ID is in the selection bitmap.
bool is_selected(uint id, StructuredBuffer<uint> selection_bitmap, uint selection_bitmap_bit_count)
{
    return id < selection_bitmap_bit_count && (selection_bitmap[id / 32] & (1u << (id % 32))) != 0;
}

// Test ray against AABB. Returns true if the ray misses (should cull).
// When the ray hits, clamps ray.t_min/t_max to the AABB intersection interval.
bool aabb_cull_ray(inout Ray ray, float3 aabb_min, float3 aabb_max, bool aabb_enabled)
{
    if (!aabb_enabled)
        return false;

    // Expand AABB to avoid precision issues causing rays that should hit to miss.
    float3 center = (aabb_min + aabb_max) * 0.5;
    float3 extent = (aabb_max - aabb_min) * 0.5;
    const float MIN_EXTENT = 1e-5;
    extent = max(extent * 1.001, float3(MIN_EXTENT));
    aabb_min = center - extent;
    aabb_max = center + extent;

    float2 near_far;
    if (!intersect_ray_aabb(ray.origin, ray.dir, aabb_min, aabb_max, near_far))
        return true;
    ray.t_min = max(ray.t_min, near_far.x);
    ray.t_max = min(ray.t_max, near_far.y);
    return false;
}

// ----------------------------------------------------------------------------
// Compute version
// ----------------------------------------------------------------------------

#ifndef __TARGET_CUDA__
/// Traces a ray per pixel using RAY_FLAG_FORCE_NON_OPAQUE.
/// Only commits hits whose geometry instance ID is in the selection bitmap,
/// effectively "seeing through" non-selected geometry.
/// Writes 1 if a selected object exists along the ray, 0 otherwise.
[shader("compute")]
[numthreads(8, 8, 1)]
public void selection_probe(
    uint2 tid: SV_DispatchThreadID,
    uniform Camera camera,
    StructuredBuffer<uint> selection_bitmap,
    uniform uint selection_bitmap_bit_count,
    RWTexture2D<uint> selected_hit_texture,
    uniform float3 aabb_min,
    uniform float3 aabb_max,
    uniform bool aabb_enabled
)
{
    uint2 dims;
    selected_hit_texture.GetDimensions(dims.x, dims.y);
    if (any(tid >= dims))
        return;

    float2 uv = (float2(tid) + 0.5) / float2(dims);
    uv.y = 1 - uv.y;

    Ray ray = camera.get_ray(uv);

    // AABB pre-test: skip trace if ray misses selection bounding box.
    if (aabb_cull_ray(ray, aabb_min, aabb_max, aabb_enabled)) {
        selected_hit_texture[tid] = 0u;
        return;
    }

    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> q;
    q.TraceRayInline(g_scene.tlas, RAY_FLAG_FORCE_NON_OPAQUE, 0xff, ray.to_ray_desc());
    while (q.Proceed()) {
        uint id = q.CandidateInstanceID() + q.CandidateGeometryIndex();
        if (is_selected(id, selection_bitmap, selection_bitmap_bit_count)) {
            q.CommitNonOpaqueTriangleHit();
            break;
        }
        // Non-selected candidates: skip (don't commit), so traversal continues behind them.
    }

    uint hit = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 1u : 0u;
    selected_hit_texture[tid] = hit;
}
#endif

// ----------------------------------------------------------------------------
// Raytracing pipeline version
// ----------------------------------------------------------------------------

// TODO: This is a workaround for CUDA not passing ray-gen entry point parameters correctly.
// https://github.com/shader-slang/slang/issues/10188
struct SelectionProbeParams
{
    Camera camera;
    StructuredBuffer<uint> selection_bitmap;
    uint selection_bitmap_bit_count;
    RWTexture2D<uint> selected_hit_texture;
    float3 aabb_min;
    float3 aabb_max;
    bool aabb_enabled;
};
ParameterBlock<SelectionProbeParams> g_selection_probe_params;

[raypayload]
public struct SelectionProbePayload
{
    uint hit : read(caller)
        : write(anyhit, miss);

    __init() { hit = 0; }
};

SCENE_RAY_TYPE_MISS(probe, probe_miss, SelectionProbePayload);
void probe_miss(inout SelectionProbePayload payload)
{
    payload.hit = 0;
}

SCENE_RAY_TYPE_ANY_HIT(probe, probe_any_hit, SelectionProbePayload);
#ifdef __TARGET_CUDA__
// Workaround for https://github.com/shader-slang/slang/issues/11658
[ForceInline]
#endif
void probe_any_hit<D : IDeferredHitInfo>(inout SelectionProbePayload payload, D deferred_hit)
{
    uint id = InstanceID() + GeometryIndex2();
    if (is_selected(
            id,
            g_selection_probe_params.selection_bitmap,
            g_selection_probe_params.selection_bitmap_bit_count
        )) {
        payload.hit = 1;
        AcceptHitAndEndSearch();
    } else {
        IgnoreHit();
    }
}

[shader("raygeneration")]
public void selection_probe_ray_gen()
{
    uint2 dims;
    g_selection_probe_params.selected_hit_texture.GetDimensions(dims.x, dims.y);

    uint2 pixel = DispatchRaysIndex().xy;
    if (any(pixel >= dims))
        return;

    float2 uv = (float2(pixel) + 0.5) / float2(dims);
    uv.y = 1 - uv.y;

    Ray ray = g_selection_probe_params.camera.get_ray(uv);

    // AABB pre-test: skip trace if ray misses selection bounding box.
    if (aabb_cull_ray(
            ray,
            g_selection_probe_params.aabb_min,
            g_selection_probe_params.aabb_max,
            g_selection_probe_params.aabb_enabled
        )) {
        g_selection_probe_params.selected_hit_texture[pixel] = 0u;
        return;
    }

    SelectionProbePayload payload = {};
    TraceRay(
        g_scene.tlas,
        RAY_FLAG_FORCE_NON_OPAQUE,
        0xff,
        0 /* RayContributionToHitGroupIndex */,
        0 /* MultiplierForGeometryContributionHitGroupIndex */,
        0 /* MissShaderIndex */,
        ray.to_ray_desc(),
        payload
    );

    g_selection_probe_params.selected_hit_texture[pixel] = payload.hit;
}
