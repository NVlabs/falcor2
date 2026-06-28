// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;

public struct LSSGeometryData
{
    public StructuredBuffer<LSSGeometryDesc> descs;
    public StructuredBuffer<PackedLSSVertex> vertices;
    public StructuredBuffer<uint> indices;
};

public extension Scene
{
    public LSSVertex get_lss_vertex(uint vertex_index)
    {
        return detail::unpack_lss_vertex(lss_geometry.vertices[vertex_index]);
    }

    /// Look up the two sphere endpoints for an LSS segment given instance/primitive IDs.
    /// @return A float2x4 where the first row is (center, radius) of the start sphere
    /// and the second row is (center, radius) of the end sphere.
    public float2x4 get_lss_segment(GeometryInstanceID geometry_instance_id, uint primitive_index)
    {
        LSSGeometryInstanceDesc instance_desc
            = reinterpret<LSSGeometryInstanceDesc>(get_geometry_instance_desc(geometry_instance_id));

        let vertex_offset = instance_desc.vertex_offset;
        let index_offset = instance_desc.index_offset + primitive_index;

        let i0 = lss_geometry.indices[index_offset];
        let i1 = i0 + 1;

        let v0 = get_lss_vertex(vertex_offset + i0);
        let v1 = get_lss_vertex(vertex_offset + i1);

        return float2x4(float4(v0.position, v0.radius), float4(v1.position, v1.radius));
    }
}

public struct LSSHitData
{
    public float2 uv;
}

/// Computes the uv parameters (from 0 to 1) from ray-curve hit position.
///
/// @param hit_pos Position of the ray-curve hit point.
/// @param sphere_a Sphere (3D position + radius) at one end point.
/// @param sphere_b Sphere (3D position + radius) at the other end point.
/// @return UV parameters.
float2 curve_hit_pos_to_uv(float3 hit_pos, float4 sphere_a, float4 sphere_b)
{
    // Build a reference frame.
    float3 fwd = sphere_b.xyz - sphere_a.xyz;
    float fwd_length = length(fwd);
    fwd /= fwd_length;
    float3 s, t;
    branchless_ONB(fwd, s, t);

    // Projection.
    float3 v = hit_pos - sphere_a.xyz;
    float z = dot(fwd, v);
    float x = dot(s, v);
    float y = dot(t, v);

    float2 result;
    // Remapping from [-r1, r2 + fwd_length] to [0, 1].
    result.x = (z + sphere_a.w) / (sphere_a.w + sphere_b.w + fwd_length);
    // Remapping from [-pi, pi] to [0, 1].
    result.y = atan2(y, x) / float.TWO_PI + 0.5;
    return result;
}

/// Computes the ray-curve hit position and the corresponding sphere center from uv parameters.
/// It is based on this paper: http://www.sci.utah.edu/publications/Han2019a/tubes-final.pdf
///
/// @param uv UV parameters of the hit point.
/// @param sphere_a Sphere (3D position + radius) at one end point.
/// @param sphere_b Sphere (3D position + radius) at the other end point.
/// @param[out] hit_pos Position of the ray-curve hit point.
/// @param[out] sphere_center Position of the corresponding sphere center (for computing the normal direction).
void uv_to_curve_hit_pos(float2 uv, float4 sphere_a, float4 sphere_b, out float3 hit_pos, out float3 sphere_center)
{
    // Build a reference frame.
    float3 fwd = sphere_b.xyz - sphere_a.xyz;
    float fwd_length = length(fwd);
    fwd /= fwd_length;
    float3 s, t;
    branchless_ONB(fwd, s, t);

    // Remapping.
    float z = uv.x * (sphere_a.w + sphere_b.w + fwd_length) - sphere_a.w;
    float phi = (uv.y - 0.5f) * float.TWO_PI;

    float p1, z1, p2, z2;
    float radius;
    float delta_z;

    if (abs(sphere_a.w - sphere_b.w) < 1e-6f) {
        p1 = z1 = 0.f;
        p2 = z2 = fwd_length;
    } else {
        p1 = fwd_length * sphere_a.w / (sphere_b.w - sphere_a.w);
        p2 = p1 + fwd_length;
        z1 = p1 - sphere_a.w * sphere_a.w / p1;
        z2 = p2 - sphere_b.w * sphere_b.w / p2;
    }

    if (p1 + z < z1) {
        // Case 1: hit_pos is on the end cap of sphere_a.
        delta_z = -z;
        radius = sphere_a.w;
    } else if (p1 + z > z2) {
        // Case 2: hit_pos is on the end cap of sphere_b.
        delta_z = fwd_length - z;
        radius = sphere_b.w;
    } else {
        // Case 3: hit_pos is on the linear segment.
        if (abs(sphere_a.w - sphere_b.w) < 1e-6f) {
            radius = sphere_a.w;
            delta_z = 0.f;
        } else {
            float ratio = (p1 + z) / z1;
            radius = sphere_a.w * ratio;
            delta_z = p1 * ratio - (p1 + z);
        }
    }

    float width = sqrt(max(radius * radius - delta_z * delta_z, 0.f));

    float3 perp_to_fwd = width * (cos(phi) * s + sin(phi) * t);
    hit_pos = sphere_a.xyz + z * fwd + perp_to_fwd;
    sphere_center = sphere_a.xyz + (z + delta_z) * fwd;
}

public HitInfo make_lss_hit_info(
    GeometryInstanceID geometry_instance_id,
    uint primitive_index,
    float3 hit_pos_os,
    float2x4 positions_and_radii_os
)
{
    HitInfo hit;
    hit.geometry_type = GeometryType::lss;
    hit.geometry_instance_id = geometry_instance_id;
    hit.primitive_index = primitive_index;
    float2 uv = curve_hit_pos_to_uv(hit_pos_os, positions_and_radii_os[0], positions_and_radii_os[1]);
    LSSHitData hit_data = { uv };
    hit._hit_data = reinterpret<HitData>(hit_data);
    return hit;
}

public HitInfo make_lss_hit_info()
{
#ifndef __TARGET_CUDA__
    float3 hit_pos_os = ObjectRayOrigin() + ObjectRayDirection() * RayTCurrent();
#else
    float3x4 world_to_object = WorldToObject3x4();
    float3 hit_pos_ws = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 hit_pos_os = transform_point(world_to_object, hit_pos_ws);
#endif
    return make_lss_hit_info(
        GeometryInstanceID(InstanceID() + GeometryIndex2()),
        PrimitiveIndex(),
        hit_pos_os,
        GetLssPositionsAndRadii()
    );
}

public extension Scene
{
    /// Construct a surface interaction from an LSS hit.
    public SurfaceInteraction make_lss_surface_interaction(HitInfo hit, float3 wi_ws)
    {
        LSSHitData hit_data = reinterpret<LSSHitData>(hit._hit_data);

        GeometryInstanceDesc geometry_instance = get_geometry_instance_desc(hit.geometry_instance_id);
        LSSGeometryInstanceDesc geometry_instance_data = reinterpret<LSSGeometryInstanceDesc>(geometry_instance);

        let vertex_offset = geometry_instance_data.vertex_offset;
        let index_offset = geometry_instance_data.index_offset + hit.primitive_index;

        let i0 = lss_geometry.indices[index_offset];
        let i1 = i0 + 1;

        let v0 = get_lss_vertex(vertex_offset + i0);
        let v1 = get_lss_vertex(vertex_offset + i1);

        float3 hit_pos;
        float3 sphere_center;
        uv_to_curve_hit_pos(
            hit_data.uv,
            float4(v0.position, v0.radius),
            float4(v1.position, v1.radius),
            hit_pos,
            sphere_center
        );

        let position_os = hit_pos;
        let normal_os = hit_pos - sphere_center;
        let tangent_os = v1.position - v0.position;
        let uv = hit_data.uv;

        let world_from_object = get_world_from_object(geometry_instance.header.transform_id);
        let inverse_transpose = transpose(float3x3(get_object_from_world(geometry_instance.header.transform_id)));

        SurfaceInteraction si = {};
        si.wi_ws = wi_ws;
        si.position_ws = transform_point(world_from_object, position_os);
        si.normal_ws = normalize(transform_vector(inverse_transpose, normal_os));
        si.front_facing = dot(wi_ws, si.normal_ws) >= 0;
        si.uv = uv;
        si.shading_frame_ws = Frame(
            si.normal_ws,
            normalize(transform_vector(world_from_object, tangent_os)),
            1.0 /*handedness*/,
            Frame::Component::tangent
        );

        return si;
    }
}

public struct DeferredLSSHitInfo : IDeferredHitInfo
{
    public HitInfo create() { return make_lss_hit_info(); }
};

// ---------------------------------------------------------------------------
// Procedural LSS support
// ---------------------------------------------------------------------------

public bool intersect_procedural_lss(
    const float3 ray_origin,
    const float3 ray_dir,
    const float t_min,
    const float t_max,
    const float2x4 segment,
    out float t
)
{
    t = {};
    float2 result;
    if (intersect_ray_linear_swept_sphere_han19(ray_origin, ray_dir, segment[0], segment[1], true, result)) {
        t = result.x;
        return t >= t_min && t <= t_max;
    }
    return false;
}

public struct ProceduralLSSIntersectionAttributes
{
    float dummy = {};
};

public struct DeferredProceduralLSSHitInfo : IDeferredHitInfo
{
    public float2x4 segment;
    public HitInfo create()
    {
#ifndef __TARGET_CUDA__
        float3 hit_pos_os = ObjectRayOrigin() + ObjectRayDirection() * RayTCurrent();
#else
        float3x4 world_to_object = WorldToObject3x4();
        float3 hit_pos_ws = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        float3 hit_pos_os = transform_point(world_to_object, hit_pos_ws);
#endif
        return make_lss_hit_info(
            GeometryInstanceID(InstanceID() + GeometryIndex2()),
            PrimitiveIndex(),
            hit_pos_os,
            segment
        );
    }
};
