// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct TriangleGeometryData
{
    public StructuredBuffer<TriangleGeometryDesc> descs;
    public StructuredBuffer<PackedTriangleVertex> vertices;
    public StructuredBuffer<uint> indices;
}

public extension Scene
{
    public TriangleVertex get_triangle_vertex(uint vertex_index)
    {
        return detail::unpack_triangle_vertex(triangle_geometry.vertices[vertex_index]);
    }

    /// Get the three vertex indices for a triangle primitive.
    public uint[3] get_triangle_indices(GeometryInstanceID geometry_instance_id, uint primitive_index)
    {
        TriangleGeometryInstanceDesc instance_desc
            = reinterpret<TriangleGeometryInstanceDesc>(get_geometry_instance_desc(geometry_instance_id));
        uint vertex_offset = instance_desc.vertex_offset;
        uint index_offset = instance_desc.index_offset + primitive_index * 3;
        return uint[3](
            vertex_offset + triangle_geometry.indices[index_offset],
            vertex_offset + triangle_geometry.indices[index_offset + 1],
            vertex_offset + triangle_geometry.indices[index_offset + 2]
        );
    }

    /// Get the three vertices for a triangle primitive.
    public TriangleVertex[3] get_triangle_vertices(GeometryInstanceID geometry_instance_id, uint primitive_index)
    {
        uint[3] vertex_indices = get_triangle_indices(geometry_instance_id, primitive_index);
        return {
            get_triangle_vertex(vertex_indices[0]),
            get_triangle_vertex(vertex_indices[1]),
            get_triangle_vertex(vertex_indices[2])
        };
    }
}

public struct TriangleHitData
{
    public float2 barycentrics;
}

public HitInfo
make_triangle_hit_info(GeometryInstanceID geometry_instance_id, uint primitive_index, float2 barycentrics)
{
    HitInfo hit;
    hit.geometry_type = GeometryType::triangle;
    hit.geometry_instance_id = geometry_instance_id;
    hit.primitive_index = primitive_index;
    TriangleHitData hit_data = { barycentrics };
    hit._hit_data = reinterpret<HitData>(hit_data);
    return hit;
}

public HitInfo make_triangle_hit_info(float2 barycentrics)
{
    return make_triangle_hit_info(GeometryInstanceID(InstanceID() + GeometryIndex2()), PrimitiveIndex(), barycentrics);
}

public extension Scene
{
    /// Construct a surface interaction from a triangle hit.
    public SurfaceInteraction make_triangle_surface_interaction(HitInfo hit, float3 wi_ws)
    {
        TriangleHitData hit_data = reinterpret<TriangleHitData>(hit._hit_data);

        GeometryInstanceDesc geometry_instance = get_geometry_instance_desc(hit.geometry_instance_id);
        TriangleGeometryInstanceDesc geometry_instance_data
            = reinterpret<TriangleGeometryInstanceDesc>(geometry_instance);

        let vertex_offset = geometry_instance_data.vertex_offset;
        let index_offset = geometry_instance_data.index_offset + hit.primitive_index * 3;

        let i0 = triangle_geometry.indices[index_offset];
        let i1 = triangle_geometry.indices[index_offset + 1];
        let i2 = triangle_geometry.indices[index_offset + 2];

        let v0 = detail::unpack_triangle_vertex(triangle_geometry.vertices[vertex_offset + i0]);
        let v1 = detail::unpack_triangle_vertex(triangle_geometry.vertices[vertex_offset + i1]);
        let v2 = detail::unpack_triangle_vertex(triangle_geometry.vertices[vertex_offset + i2]);

        // barycentric weights
        let bw = float3(
            1.0 - hit_data.barycentrics.x - hit_data.barycentrics.y,
            hit_data.barycentrics.x,
            hit_data.barycentrics.y
        );

        let position_os = bw.x * v0.position + bw.y * v1.position + bw.z * v2.position;
        let shading_normal_os = bw.x * v0.normal + bw.y * v1.normal + bw.z * v2.normal;
        let shading_tangent_os = bw.x * v0.tangent + bw.y * v1.tangent + bw.z * v2.tangent;
        let uv = bw.x * v0.uv[0] + bw.y * v1.uv[0] + bw.z * v2.uv[0];

        let is_front_face_cw = false;
        let N = normalize(cross(v1.position - v0.position, v2.position - v0.position));
        let normal_os = select(is_front_face_cw, -N, N);
        let tangent_os = normalize(v1.position - v0.position);

        let world_from_object = get_world_from_object(geometry_instance.header.transform_id);
        let inverse_transpose = transpose(float3x3(get_object_from_world(geometry_instance.header.transform_id)));
        // All handedness values should be the same for the three vertices of a triangle, so we can just use v0's here.
        let transform_handedness = determinant(float3x3(world_from_object)) < 0.0 ? -1.0 : 1.0;
        let shading_handedness = v0.handedness * transform_handedness;

        SurfaceInteraction si = {};
        si.wi_ws = wi_ws;
        si.position_ws = transform_point(world_from_object, position_os);
        si.normal_ws = normalize(transform_vector(inverse_transpose, normal_os));
        si.front_facing = dot(wi_ws, si.normal_ws) >= 0;
        si.uv = uv;
        si.shading_frame_ws = Frame(
            normalize(transform_vector(inverse_transpose, shading_normal_os)),
            normalize(transform_vector(world_from_object, shading_tangent_os)),
            shading_handedness
        );
        return si;
    }
}

public struct DeferredTriangleHitInfo : IDeferredHitInfo
{
    public float2 barycentrics;
    public HitInfo create() { return make_triangle_hit_info(barycentrics); }
};
