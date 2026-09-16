// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;
import sgl.device.print;
import sgl.device.nvapi;

public interface ISceneIntersector
{
    public IHitInfo intersect(Ray ray);

    public bool is_occluded(Ray ray);
};

/// Opacity evaluators are stored directly in ray payloads. Implementations must be
/// tiny, copyable value types and should only contain state required for one ray.
public interface IOpacityEvaluator
{
    public static const uint RAY_FLAGS;

    public __init();
    public __init(uint stochastic_seed);

    public bool is_valid_hit(Ray ray, IHitInfo hit, MaterialInstanceHints hints);
}

public struct AcceptAllHitsEvaluator : IOpacityEvaluator
{
    public static const uint RAY_FLAGS = RAY_FLAG_FORCE_OPAQUE;

    public __init() { }
    public __init(uint stochastic_seed) { }

    public bool is_valid_hit(Ray ray, IHitInfo hit, MaterialInstanceHints hints) { return true; }
};

public struct MaterialOpacityEvaluator : IOpacityEvaluator
{
    public static const uint RAY_FLAGS = 0;

    uint stochastic_seed;

    public __init() { stochastic_seed = 0; }
    public __init(uint stochastic_seed) { this.stochastic_seed = stochastic_seed; }

    public bool is_valid_hit(Ray ray, IHitInfo hit, MaterialInstanceHints hints)
    {
        // TODO(@skallweit): Non-triangle opacity is intentionally unsupported. Adding LSS support requires defining
        // its opacity semantics, including texture UV mapping, marking hardware and procedural LSS instances
        // non-opaque, and evaluating procedural ray-query candidates before committing them.
        if (hit.geometry_type != GeometryType::triangle)
            return true;

        MaterialHeader header = g_scene.get_material_header(hit);
        OpacityFlags opacity_flags = header.opacity_flags;
        if ((opacity_flags & OpacityFlags::enabled) == OpacityFlags::none)
            return true;

        float opacity;
        if ((opacity_flags & OpacityFlags::evaluate_material) != OpacityFlags::none) {
            SurfaceInteraction si = g_scene.make_surface_interaction(hit, -ray.dir);
            IMaterial material = g_scene.get_material(hit);
            opacity = material.eval_opacity(si, ExplicitLodSampler(0), hints);
        } else {
            opacity = header.opacity_factor;
            TextureHandle<float4> opacity_texture = TextureHandle<float4>(header.opacity_texture);
            if (opacity_texture.is_valid()) {
                SurfaceInteraction si = g_scene.make_surface_interaction(hit, -ray.dir);
                opacity *= opacity_texture
                               .sample(ExplicitLodSampler(0), si.uv, float4(1.0))[header.opacity_texture_channel];
            }
        }
        opacity = saturate(opacity);

        if ((opacity_flags & OpacityFlags::use_threshold) != OpacityFlags::none)
            return opacity >= header.opacity_threshold;

        uint hash = jenkins_hash(stochastic_seed ^ uint(hit.geometry_instance_id));
        hash = jenkins_hash(hash ^ hit.primitive_index);
        float sample = (hash >> 8) * 0x1p-24;
        return sample < opacity;
    }
};

public extern struct OpacityEvaluator : IOpacityEvaluator = AcceptAllHitsEvaluator;


public struct RayQuerySceneIntersector : ISceneIntersector
{
    OpacityEvaluator opacity_evaluator;

    public __init() { opacity_evaluator = OpacityEvaluator(0); }
    public __init(OpacityEvaluator opacity_evaluator) { this.opacity_evaluator = opacity_evaluator; }

    public IHitInfo intersect(Ray ray) { return intersect(ray, MaterialInstanceHints::none); }

    public IHitInfo intersect(Ray ray, MaterialInstanceHints hints)
    {
        RayQuery<OpacityEvaluator.RAY_FLAGS> q;
        q.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
        while (q.Proceed()) {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                TriangleHitInfo temp_hit = TriangleHitInfo(
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()),
                    q.CandidatePrimitiveIndex(),
                    q.CandidateTriangleBarycentrics()
                );

                if (!opacity_evaluator.is_valid_hit(ray, temp_hit, hints))
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
                LSSHitInfo temp_hit = LSSHitInfo(
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()),
                    q.CandidatePrimitiveIndex(),
                    hit_pos_os,
                    q.CandidateLssPositionsAndRadii()
                );

                if (!opacity_evaluator.is_valid_hit(ray, temp_hit, hints))
                    continue;

                q.CommitNonOpaqueBuiltinPrimitiveHit();
            }
        };

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            return TriangleHitInfo(
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
                return LSSHitInfo(
                    geometry_instance_id,
                    q.CommittedPrimitiveIndex(),
                    hit_pos_os,
                    g_scene.get_lss_segment(geometry_instance_id, q.CommittedPrimitiveIndex())
                );
            }
            default:
                break;
            }
        } else if (q.CommittedIsLss()) {
            float3 hit_pos_os = q.CommittedObjectRayOrigin() + q.CommittedObjectRayDirection() * q.CommittedRayT();
            return LSSHitInfo(
                GeometryInstanceID(q.CommittedInstanceID() + q.CommittedGeometryIndex()),
                q.CommittedPrimitiveIndex(),
                hit_pos_os,
                q.CommittedLssPositionsAndRadii()
            );
        }
        return InvalidHitInfo();
    }

    public bool is_occluded(Ray ray) { return is_occluded(ray, MaterialInstanceHints::none); }

    public bool is_occluded(Ray ray, MaterialInstanceHints hints)
    {
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | OpacityEvaluator.RAY_FLAGS> q;
        q.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
        while (q.Proceed()) {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                TriangleHitInfo temp_hit = TriangleHitInfo(
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()),
                    q.CandidatePrimitiveIndex(),
                    q.CandidateTriangleBarycentrics()
                );

                if (!opacity_evaluator.is_valid_hit(ray, temp_hit, hints))
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
                LSSHitInfo temp_hit = LSSHitInfo(
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()),
                    q.CandidatePrimitiveIndex(),
                    hit_pos_os,
                    q.CandidateLssPositionsAndRadii()
                );

                if (!opacity_evaluator.is_valid_hit(ray, temp_hit, hints))
                    continue;

                q.CommitNonOpaqueBuiltinPrimitiveHit();
            }
        };
        return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT
            || q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT || q.CommittedIsLss();
    }
};
