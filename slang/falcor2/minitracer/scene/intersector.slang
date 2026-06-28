// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing scene;

public interface ISceneIntersector
{
    public HitInfo intersect(Ray ray, float alpha_threshold = 1.0);

    public bool is_occluded(Ray ray, float alpha_threshold = 1.0);
};

public struct RayQuerySceneIntersector : ISceneIntersector
{
    public HitInfo intersect(Ray ray, float alpha_threshold)
    {
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
        while (q.Proceed()) {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                HitInfo temp_hit = {
                    true,
                    q.CandidateTriangleBarycentrics(),
                    q.CandidateTriangleRayT(),
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()),
                    q.CandidatePrimitiveIndex(),
                };
                if (!g_scene.is_valid_hit(temp_hit, ray.dir, alpha_threshold))
                    continue;
                q.CommitNonOpaqueTriangleHit();
            }
        };

        HitInfo hit = {};
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            hit = {
                true,
                q.CommittedTriangleBarycentrics(),
                q.CommittedRayT(),
                GeometryInstanceID(q.CommittedInstanceID() + q.CommittedGeometryIndex()),
                q.CommittedPrimitiveIndex(),
            };
        }
        return hit;
    }

    public bool is_occluded(Ray ray, float alpha_threshold)
    {
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
        while (q.Proceed()) {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                HitInfo temp_hit = {
                    true,
                    q.CandidateTriangleBarycentrics(),
                    q.CandidateTriangleRayT(),
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()),
                    q.CandidatePrimitiveIndex(),
                };
                if (!g_scene.is_valid_hit(temp_hit, ray.dir, alpha_threshold))
                    continue;
                q.CommitNonOpaqueTriangleHit();
            }
        };
        return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    }
};
