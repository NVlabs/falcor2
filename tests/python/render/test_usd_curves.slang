// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

public struct CurveUVProbe
{
    public bool valid;
    public float2 curve_parameters;
    public float2 uv;
    public float2 uv0;
    public float2 uv1;
    public float3 position;
    public float3 normal;
    public Frame frame;
    public uint additional_mask;
};

CurveUVProbe read_curve_hit(LSSHitInfo hit)
{
    CurveUVProbe result = {};
    result.valid = true;
    result.curve_parameters = hit.curve_parameters;
    const SurfaceInteraction si = g_scene.make_lss_surface_interaction(hit, float3(0, 0, 1));
    result.uv = si.uv;
    result.position = si.position_ws;
    result.normal = si.normal_ws;
    result.frame = si.shading_frame_ws;
    const LSSGeometryInstanceDesc instance
        = reinterpret<LSSGeometryInstanceDesc>(g_scene.get_geometry_instance_desc(hit.geometry_instance_id));
    const uint index
        = g_scene.geometry[instance.index_buffer].as_buffer<uint>()[instance.index_offset + hit.primitive_index]
        + instance.vertex_offset;
    result.uv0 = g_scene.get_lss_vertex(instance.vertex_buffer, index).texcoord;
    result.uv1 = g_scene.get_lss_vertex(instance.vertex_buffer, index + 1).texcoord;
    const PackedAdditionalAttribute header
        = g_scene.lss_geometry
              .additional_attributes[uint(instance.header.geometry_id) * ADDITIONAL_ATTRIBUTE_DESC_ROW_SIZE];
    result.additional_mask = header.byte_offset;
    return result;
}

public CurveUVProbe curve_uv_probe(uint geometry_instance_id, uint primitive_index, float2 curve_parameters)
{
    PackedHitInfo packed = {};
    packed.geometry_type = GeometryType::lss;
    packed.geometry_instance_id = GeometryInstanceID(geometry_instance_id);
    packed.payload[0] = primitive_index;
    packed.payload[1] = asuint(curve_parameters.x);
    packed.payload[2] = asuint(curve_parameters.y);
    return read_curve_hit(LSSHitInfo(packed));
}

public CurveUVProbe curve_ray_probe(float3 origin)
{
    RayQuerySceneIntersector intersector = {};
    const IHitInfo hit = intersector.intersect(Ray(origin, float3(0, 0, -1)));
    if (hit.valid && hit.geometry_type == GeometryType::lss)
        return read_curve_hit(LSSHitInfo(hit.pack()));
    return {};
}

// Exercise the software intersection explicitly even on devices with native LSS support.
// Constructing the hit from its position also tests forward parameter conversion and reconstruction.
public CurveUVProbe curve_procedural_probe(uint geometry_instance_id, float3 origin)
{
    const GeometryInstanceID id = GeometryInstanceID(geometry_instance_id);
    const float2x4 segment = g_scene.get_lss_segment(id, 0);
    float t;
    const float3 direction = float3(0, 0, -1);
    if (intersect_procedural_lss(origin, direction, 0.0, float.MAX, segment, t))
        return read_curve_hit(LSSHitInfo(id, 0, origin + t * direction, segment));
    return {};
}
