// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

public float3 object_origin(uint geometry_instance_id)
{
    const TriangleHitInfo hit = TriangleHitInfo(GeometryInstanceID(geometry_instance_id), 0, float2(0.2, 0.3));
    const SurfaceInteraction si = g_scene.make_surface_interaction(hit, float3(0.0, 0.0, 1.0));
    return transform_point(g_scene.get_geometry_world_from_object(si), float3(0.0));
}

public uint nonopaque_triangle_count()
{
    RayDesc ray;
    ray.Origin = float3(0.2, 0.1, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 2.0;
    RayQuery<0> query;
    query.TraceRayInline(g_scene.tlas, 0, 0xff, ray);
    uint count = 0;
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            ++count;
            query.CommitNonOpaqueTriangleHit();
        }
    }
    return count;
}
