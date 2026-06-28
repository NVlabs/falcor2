// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import sgl.device.print;
import sgl.device.nvapi;

public interface ISceneIntersector
{
    public HitInfo intersect(Ray ray);

    public bool is_occluded(Ray ray);
};

public interface IOpacityEvaluator
{
    public static bool is_valid_hit(Ray ray, HitInfo hit);
}

public struct AcceptAllHitsEvaluator : IOpacityEvaluator
{
    public static bool is_valid_hit(Ray ray, HitInfo hit) { return true; }
};

extern struct OpacityEvaluator : IOpacityEvaluator = AcceptAllHitsEvaluator;


public struct RayQuerySceneIntersector : ISceneIntersector
{
    public HitInfo intersect(Ray ray)
    {
        RayQuery<> q;
        q.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
        while (q.Proceed()) {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                HitInfo temp_hit = make_triangle_hit_info(
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()),
                    q.CandidatePrimitiveIndex(),
                    q.CandidateTriangleBarycentrics()
                );

                if (!OpacityEvaluator::is_valid_hit(ray, temp_hit))
                    continue;

                q.CommitNonOpaqueTriangleHit();
            } else if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                GeometryInstanceID geometry_instance_id
                    = GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex());
                GeometryType geometry_type = g_scene.get_geometry_instance_type(geometry_instance_id);
                switch (geometry_type) {
                case GeometryType::lss: {
                    float t = {};
                    if (intersect_procedural_lss(
                            q.CandidateObjectRayOrigin(),
                            q.CandidateObjectRayDirection(),
                            q.RayTMin(),
                            q.CommittedRayT(),
                            g_scene.get_lss_segment(geometry_instance_id, q.CandidatePrimitiveIndex()),
                            t
                        )) {
                        q.CommitProceduralPrimitiveHit(t);
                    }
                    continue;
                }
                default:
                    continue;
                }
            } else if (q.CandidateIsNonOpaqueLss()) {
                float3 hit_pos_os = q.CandidateObjectRayOrigin()
                    + q.CandidateObjectRayDirection() * q.CandidateBuiltinPrimitiveRayT();
                HitInfo temp_hit = make_lss_hit_info(
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()),
                    q.CandidatePrimitiveIndex(),
                    hit_pos_os,
                    q.CandidateLssPositionsAndRadii()
                );

                if (!OpacityEvaluator::is_valid_hit(ray, temp_hit))
                    continue;

                q.CommitNonOpaqueBuiltinPrimitiveHit();
            }
        };

        HitInfo hit = {};
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            hit = make_triangle_hit_info(
                GeometryInstanceID(q.CommittedInstanceID() + q.CommittedGeometryIndex()),
                q.CommittedPrimitiveIndex(),
                q.CommittedTriangleBarycentrics()
            );
        } else if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            GeometryInstanceID geometry_instance_id
                = GeometryInstanceID(q.CommittedInstanceID() + q.CommittedGeometryIndex());
            GeometryType geometry_type = g_scene.get_geometry_instance_type(geometry_instance_id);
            switch (geometry_type) {
            case GeometryType::lss: {
                float3 hit_pos_os = q.CommittedObjectRayOrigin() + q.CommittedObjectRayDirection() * q.CommittedRayT();
                hit = make_lss_hit_info(
                    geometry_instance_id,
                    q.CommittedPrimitiveIndex(),
                    hit_pos_os,
                    g_scene.get_lss_segment(geometry_instance_id, q.CommittedPrimitiveIndex())
                );
                break;
            }
            default:
                break;
            }
        } else if (q.CommittedIsLss()) {
            float3 hit_pos_os = q.CommittedObjectRayOrigin() + q.CommittedObjectRayDirection() * q.CommittedRayT();
            hit = make_lss_hit_info(
                GeometryInstanceID(q.CommittedInstanceID() + q.CommittedGeometryIndex()),
                q.CommittedPrimitiveIndex(),
                hit_pos_os,
                q.CommittedLssPositionsAndRadii()
            );
        }
        return hit;
    }

    public bool is_occluded(Ray ray)
    {
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        q.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
        while (q.Proceed()) {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                HitInfo temp_hit = make_triangle_hit_info(
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()),
                    q.CandidatePrimitiveIndex(),
                    q.CandidateTriangleBarycentrics()
                );

                if (!OpacityEvaluator::is_valid_hit(ray, temp_hit))
                    continue;

                q.CommitNonOpaqueTriangleHit();
            } else if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                GeometryInstanceID geometry_instance_id
                    = GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex());
                GeometryType geometry_type = g_scene.get_geometry_instance_type(geometry_instance_id);
                switch (geometry_type) {
                case GeometryType::lss: {
                    float t = {};
                    if (intersect_procedural_lss(
                            q.CandidateObjectRayOrigin(),
                            q.CandidateObjectRayDirection(),
                            q.RayTMin(),
                            q.CommittedRayT(),
                            g_scene.get_lss_segment(geometry_instance_id, q.CandidatePrimitiveIndex()),
                            t
                        )) {
                        q.CommitProceduralPrimitiveHit(t);
                    }
                    continue;
                }
                default:
                    continue;
                }
            } else if (q.CandidateIsNonOpaqueLss()) {
                float3 hit_pos_os = q.CandidateObjectRayOrigin()
                    + q.CandidateObjectRayDirection() * q.CandidateBuiltinPrimitiveRayT();
                HitInfo temp_hit = make_lss_hit_info(
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()),
                    q.CandidatePrimitiveIndex(),
                    hit_pos_os,
                    q.CandidateLssPositionsAndRadii()
                );

                if (!OpacityEvaluator::is_valid_hit(ray, temp_hit))
                    continue;

                q.CommitNonOpaqueBuiltinPrimitiveHit();
            }
        };
        return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT
            || q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT || q.CommittedIsLss();
    }
};
