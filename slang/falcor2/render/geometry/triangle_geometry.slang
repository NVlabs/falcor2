// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct TriangleGeometryData
{
    public StructuredBuffer<TriangleGeometryDesc> descs;
    public StructuredBuffer<PackedAdditionalAttribute> additional_attributes;
    public ByteAddressBuffer additional_attribute_data;
}

public extension Scene
{
    public TriangleVertex get_triangle_vertex(uint buffer_index, uint vertex_index)
    {
        return detail::unpack_triangle_vertex(geometry[buffer_index].as_buffer<PackedTriangleVertex>()[vertex_index]);
    }

    /// Get vertex indices within the instance's selected vertex buffer.
    public uint[3] get_triangle_indices(GeometryInstanceID geometry_instance_id, uint primitive_index)
    {
        TriangleGeometryInstanceDesc instance_desc
            = reinterpret<TriangleGeometryInstanceDesc>(get_geometry_instance_desc(geometry_instance_id));
        uint vertex_offset = instance_desc.vertex_offset;
        uint index_offset = instance_desc.index_offset + primitive_index * 3;
        let indices = geometry[instance_desc.index_buffer].as_buffer<uint>();
        return uint[3](
            vertex_offset + indices[index_offset],
            vertex_offset + indices[index_offset + 1],
            vertex_offset + indices[index_offset + 2]
        );
    }

    /// Get the three vertices for a triangle primitive.
    public TriangleVertex[3] get_triangle_vertices(GeometryInstanceID geometry_instance_id, uint primitive_index)
    {
        TriangleGeometryInstanceDesc instance_desc
            = reinterpret<TriangleGeometryInstanceDesc>(get_geometry_instance_desc(geometry_instance_id));
        uint[3] vertex_indices = get_triangle_indices(geometry_instance_id, primitive_index);
        let vertices = geometry[instance_desc.vertex_buffer].as_buffer<PackedTriangleVertex>();
        return {
            detail::unpack_triangle_vertex(vertices[vertex_indices[0]]),
            detail::unpack_triangle_vertex(vertices[vertex_indices[1]]),
            detail::unpack_triangle_vertex(vertices[vertex_indices[2]])
        };
    }

    /// Previous vertices occupy vertex_buffer + 1 and share the current topology and vertex offsets.
    /// Requires vertex_motion and a two-buffer vertex tuple with previous-frame data populated.
    public float3 get_previous_position_os(TriangleHitInfo hit)
    {
        TriangleGeometryInstanceDesc instance_desc
            = reinterpret<TriangleGeometryInstanceDesc>(get_geometry_instance_desc(hit.geometry_instance_id));
        let vertex_indices = get_triangle_indices(hit.geometry_instance_id, hit.primitive_index);
        let vertices = geometry[instance_desc.vertex_buffer + 1].as_buffer<PackedTriangleVertex>();
        let v0 = detail::unpack_triangle_vertex(vertices[vertex_indices[0]]);
        let v1 = detail::unpack_triangle_vertex(vertices[vertex_indices[1]]);
        let v2 = detail::unpack_triangle_vertex(vertices[vertex_indices[2]]);
        let weights = float3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics);
        return weights.x * v0.position + weights.y * v1.position + weights.z * v2.position;
    }
}

public struct TriangleHitInfo : IHitInfo
{
    public property GeometryType geometry_type
    {
        get {
            return GeometryType::triangle;
        }
    }

    public GeometryInstanceID geometry_instance_id;

    public uint primitive_index;

    public property bool valid
    {
        get {
            return true;
        }
    }

    public float2 barycentrics;

    public __init(GeometryInstanceID geometry_instance_id, uint primitive_index, float2 barycentrics)
    {
        this.geometry_instance_id = geometry_instance_id;
        this.primitive_index = primitive_index;
        this.barycentrics = barycentrics;
    }

    public __init(const PackedHitInfo p)
    {
        geometry_instance_id = p.geometry_instance_id;
        primitive_index = p.payload[0];
        barycentrics = float2(asfloat(p.payload[1]), asfloat(p.payload[2]));
    }

    public PackedHitInfo pack()
    {
        PackedHitInfo p = {};
        p.geometry_type = geometry_type;
        p.geometry_instance_id = geometry_instance_id;
        p.payload[0] = primitive_index;
        p.payload[1] = asuint(barycentrics.x);
        p.payload[2] = asuint(barycentrics.y);
        return p;
    }
};

public struct TriangleGeomPropAccessor : IGeomPropAccessor
{
    public no_diff GeometryInstanceID geometry_instance_id;
    public no_diff uint additional_attribute_desc_offset;
    public no_diff uint additional_attribute_mask;
    public no_diff uint3 local_vertex_indices;
    public no_diff float3 barycentric_weights;

    public __init(const TriangleHitInfo hit)
    {
        geometry_instance_id = hit.geometry_instance_id;
        const GeometryInstanceDesc geometry_instance = g_scene.get_geometry_instance_desc(geometry_instance_id);
        const TriangleGeometryInstanceDesc triangle_geometry_instance
            = reinterpret<TriangleGeometryInstanceDesc>(geometry_instance);
        additional_attribute_desc_offset
            = uint(triangle_geometry_instance.header.geometry_id) * ADDITIONAL_ATTRIBUTE_DESC_ROW_SIZE;
        additional_attribute_mask
            = g_scene.triangle_geometry.additional_attributes[additional_attribute_desc_offset].byte_offset;
        const uint index_offset = triangle_geometry_instance.index_offset + hit.primitive_index * 3;
        let indices = g_scene.geometry[triangle_geometry_instance.index_buffer].as_buffer<uint>();
        local_vertex_indices = uint3(indices[index_offset], indices[index_offset + 1], indices[index_offset + 2]);
        barycentric_weights
            = float3(1.f - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x, hit.barycentrics.y);
    }

    public bool has_attribute(uint id) { return id < 32 && (additional_attribute_mask & (1u << id)) != 0; }

    bool try_get_attribute_desc(
        uint attribute_id,
        out PackedAdditionalAttribute attribute,
        out AdditionalAttributeType type,
        out uint component_count
    )
    {
        attribute = {};
        type = AdditionalAttributeType::invalid;
        component_count = 0;
        if (attribute_id >= 32)
            return false;

        const uint attribute_bit = 1u << attribute_id;
        if ((additional_attribute_mask & attribute_bit) == 0)
            return false;

        const uint attribute_rank = countbits(additional_attribute_mask & (attribute_bit - 1u));
        attribute
            = g_scene.triangle_geometry.additional_attributes[additional_attribute_desc_offset + 1 + attribute_rank];
        component_count = attribute.format & 0xff;
        type = AdditionalAttributeType(attribute.format >> 8);
        return true;
    }

    uint get_dominant_local_vertex_index()
    {
        uint dominant_vertex = 0;
        float dominant_weight = barycentric_weights.x;
        if (barycentric_weights.y > dominant_weight) {
            dominant_vertex = 1;
            dominant_weight = barycentric_weights.y;
        }
        if (barycentric_weights.z > dominant_weight)
            dominant_vertex = 2;
        return local_vertex_indices[dominant_vertex];
    }

    bool try_load_attribute<let N : int>(uint attribute_id, out vector<float, N> value)
    {
        value = vector<float, N>(0.f);

        PackedAdditionalAttribute attribute;
        AdditionalAttributeType type;
        uint component_count;
        if (!try_get_attribute_desc(attribute_id, attribute, type, component_count))
            return false;
        if (type != AdditionalAttributeType::float16 && type != AdditionalAttributeType::float32)
            return false;

        // This is a bit ugly and long, but is the most efficient way to load all vertex
        // values without extra branching.
        float4 loaded_values[3];
        [ForceUnroll]
        for (uint vertex = 0; vertex < 3; ++vertex)
            loaded_values[vertex] = 0.f;

        if (type == AdditionalAttributeType::float16) {
            const uint3 addresses = attribute.byte_offset + local_vertex_indices * component_count * sizeof(float16_t);
            switch (component_count) {
            case 1:
                [ForceUnroll] for (uint vertex = 0; vertex < 3; ++vertex) loaded_values[vertex].x = float(
                    g_scene.triangle_geometry.additional_attribute_data.LoadAligned<float16_t>(addresses[vertex])
                );
                break;
            case 2:
                [ForceUnroll] for (uint vertex = 0; vertex < 3; ++vertex) loaded_values[vertex].xy = float2(
                    g_scene.triangle_geometry.additional_attribute_data.LoadAligned<float16_t2>(addresses[vertex])
                );
                break;
            case 3:
                [ForceUnroll] for (uint vertex = 0; vertex < 3; ++vertex) loaded_values[vertex].xyz = float3(
                    g_scene.triangle_geometry.additional_attribute_data.LoadAligned<float16_t3>(addresses[vertex])
                );
                break;
            case 4:
                [ForceUnroll] for (uint vertex = 0; vertex < 3; ++vertex) loaded_values[vertex] = float4(
                    g_scene.triangle_geometry.additional_attribute_data.LoadAligned<float16_t4>(addresses[vertex])
                );
                break;
            }
        } else {
            const uint3 addresses = attribute.byte_offset + local_vertex_indices * component_count * sizeof(float);
            switch (component_count) {
            case 1:
                [ForceUnroll] for (uint vertex = 0; vertex < 3; ++vertex) loaded_values[vertex].x
                    = g_scene.triangle_geometry.additional_attribute_data.LoadAligned<float>(addresses[vertex]);
                break;
            case 2:
                [ForceUnroll] for (uint vertex = 0; vertex < 3; ++vertex) loaded_values[vertex].xy
                    = g_scene.triangle_geometry.additional_attribute_data.LoadAligned<float2>(addresses[vertex]);
                break;
            case 3:
                [ForceUnroll] for (uint vertex = 0; vertex < 3; ++vertex) loaded_values[vertex].xyz
                    = g_scene.triangle_geometry.additional_attribute_data.LoadAligned<float3>(addresses[vertex]);
                break;
            case 4:
                [ForceUnroll] for (uint vertex = 0; vertex < 3; ++vertex) loaded_values[vertex]
                    = g_scene.triangle_geometry.additional_attribute_data.LoadAligned<float4>(addresses[vertex]);
                break;
            }
        }

        const float4 interpolated_value = barycentric_weights.x * loaded_values[0]
            + barycentric_weights.y * loaded_values[1] + barycentric_weights.z * loaded_values[2];
        [ForceUnroll]
        for (uint component = 0; component < N; ++component)
            value[component] = interpolated_value[component];
        return true;
    }

    bool try_load_int_attribute<let N : int>(uint attribute_id, out vector<int, N> value)
    {
        value = vector<int, N>(0);
        PackedAdditionalAttribute attribute;
        AdditionalAttributeType type;
        uint component_count;
        if (!try_get_attribute_desc(attribute_id, attribute, type, component_count)
            || type != AdditionalAttributeType::int32)
            return false;

        const uint address = attribute.byte_offset + get_dominant_local_vertex_index() * component_count * sizeof(int);
        int4 loaded_value = 0;
        switch (component_count) {
        case 1:
            loaded_value.x = g_scene.triangle_geometry.additional_attribute_data.LoadAligned<int>(address);
            break;
        case 2:
            loaded_value.xy = g_scene.triangle_geometry.additional_attribute_data.LoadAligned<int2>(address);
            break;
        case 3:
            loaded_value.xyz = g_scene.triangle_geometry.additional_attribute_data.LoadAligned<int3>(address);
            break;
        case 4:
            loaded_value = g_scene.triangle_geometry.additional_attribute_data.LoadAligned<int4>(address);
            break;
        }
        [ForceUnroll]
        for (uint component = 0; component < N; ++component)
            value[component] = loaded_value[component];
        return true;
    }

    uint load_packed_bool_attribute<let ComponentCount : int>(uint address)
    {
        const uint aligned_address = address & ~3u;
        const uint byte_offset = address & 3u;
        uint value = g_scene.triangle_geometry.additional_attribute_data.LoadAligned<uint>(aligned_address)
            >> (byte_offset * 8u);
        if (byte_offset + uint(ComponentCount) > 4u) {
            value |= g_scene.triangle_geometry.additional_attribute_data.LoadAligned<uint>(aligned_address + 4u)
                << ((4u - byte_offset) * 8u);
        }
        return value;
    }

    bool try_load_bool_attribute<let N : int>(uint attribute_id, out vector<bool, N> value)
    {
        value = vector<bool, N>(false);
        PackedAdditionalAttribute attribute;
        AdditionalAttributeType type;
        uint component_count;
        if (!try_get_attribute_desc(attribute_id, attribute, type, component_count)
            || type != AdditionalAttributeType::bool_)
            return false;

        const uint address = attribute.byte_offset + get_dominant_local_vertex_index() * component_count;
        bool4 loaded_value = false;
        switch (component_count) {
        case 1: {
            const uint packed_value = load_packed_bool_attribute<1>(address);
            loaded_value.x = (packed_value & 0xffu) != 0;
            break;
        }
        case 2: {
            const uint packed_value = load_packed_bool_attribute<2>(address);
            loaded_value.xy = bool2((packed_value & 0xffu) != 0, (packed_value & 0xff00u) != 0);
            break;
        }
        case 3: {
            const uint packed_value = load_packed_bool_attribute<3>(address);
            loaded_value.xyz
                = bool3((packed_value & 0xffu) != 0, (packed_value & 0xff00u) != 0, (packed_value & 0xff0000u) != 0);
            break;
        }
        case 4: {
            // Four-byte elements are naturally aligned because the stream starts at a 16-byte boundary.
            const uint packed_value = g_scene.triangle_geometry.additional_attribute_data.LoadAligned<uint>(address);
            loaded_value = bool4(
                (packed_value & 0xffu) != 0,
                (packed_value & 0xff00u) != 0,
                (packed_value & 0xff0000u) != 0,
                (packed_value & 0xff000000u) != 0
            );
            break;
        }
        }
        [ForceUnroll]
        for (uint component = 0; component < N; ++component)
            value[component] = loaded_value[component];
        return true;
    }

    public bool try_get(uint attribute_id, out int value)
    {
        value = 0;
        vector<int, 1> loaded_value;
        if (!try_load_int_attribute(attribute_id, loaded_value))
            return false;
        value = loaded_value.x;
        return true;
    }

    public bool try_get(uint attribute_id, out int2 value) { return try_load_int_attribute(attribute_id, value); }

    public bool try_get(uint attribute_id, out int3 value) { return try_load_int_attribute(attribute_id, value); }

    public bool try_get(uint attribute_id, out int4 value) { return try_load_int_attribute(attribute_id, value); }

    public bool try_get(uint attribute_id, out bool value)
    {
        value = false;
        vector<bool, 1> loaded_value;
        if (!try_load_bool_attribute(attribute_id, loaded_value))
            return false;
        value = loaded_value.x;
        return true;
    }

    public bool try_get(uint attribute_id, out bool2 value) { return try_load_bool_attribute(attribute_id, value); }

    public bool try_get(uint attribute_id, out bool3 value) { return try_load_bool_attribute(attribute_id, value); }

    public bool try_get(uint attribute_id, out bool4 value) { return try_load_bool_attribute(attribute_id, value); }

    public bool try_get(uint attribute_id, out float value)
    {
        value = 0.f;
        vector<float, 1> loaded_value;
        if (!try_load_attribute(attribute_id, loaded_value))
            return false;
        value = loaded_value.x;
        return true;
    }

    public bool try_get(uint attribute_id, out float2 value) { return try_load_attribute(attribute_id, value); }

    public bool try_get(uint attribute_id, out float3 value) { return try_load_attribute(attribute_id, value); }

    public bool try_get(uint attribute_id, out float4 value) { return try_load_attribute(attribute_id, value); }

    public bool try_get_additional_texcoord(uint index, out float2 value)
    {
        value = 0.f;
        return index > 0 && index < 8 && try_get(index, value);
    }

    public bool try_get_additional_texcoord(uint index, out float3 value)
    {
        value = 0.f;
        return index < 8 && try_get(index, value);
    }

    public bool try_get_additional_texcoord(uint index, out float4 value)
    {
        value = 0.f;
        return index < 8 && try_get(index, value);
    }

    public bool try_get_color(uint index, out float4 value)
    {
        value = 0.f;
        if (index != 0)
            return false;

        PackedAdditionalAttribute attribute;
        AdditionalAttributeType type;
        uint component_count;
        if (!try_get_attribute_desc(GEOMETRY_ATTRIBUTE_ID_COLOR_0, attribute, type, component_count)
            || (type != AdditionalAttributeType::float16 && type != AdditionalAttributeType::float32)
            || (component_count != 3 && component_count != 4))
            return false;

        if (!try_get(GEOMETRY_ATTRIBUTE_ID_COLOR_0, value))
            return false;
        if (component_count == 3)
            value.w = 1.f;
        return true;
    }

    public float4x4 get_space_transform(GeomPropSpace to_space, GeomPropSpace from_space)
    {
        if (to_space == from_space || (to_space != GeomPropSpace::world && from_space != GeomPropSpace::world))
            return geomprop_identity4x4();
        return to_space == GeomPropSpace::world ? g_scene.get_world_from_object(geometry_instance_id)
                                                : g_scene.get_object_from_world(geometry_instance_id);
    }
};

public extension Scene
{
    /// Construct a surface interaction from a triangle hit.
    public SurfaceInteraction make_triangle_surface_interaction(TriangleHitInfo hit, float3 wi_ws)
    {
        GeometryInstanceDesc geometry_instance = get_geometry_instance_desc(hit.geometry_instance_id);
        TriangleGeometryInstanceDesc geometry_instance_data
            = reinterpret<TriangleGeometryInstanceDesc>(geometry_instance);

        let vertex_offset = geometry_instance_data.vertex_offset;
        let index_offset = geometry_instance_data.index_offset + hit.primitive_index * 3;

        let indices = geometry[geometry_instance_data.index_buffer].as_buffer<uint>();
        let vertices = geometry[geometry_instance_data.vertex_buffer].as_buffer<PackedTriangleVertex>();

        let i0 = indices[index_offset];
        let i1 = indices[index_offset + 1];
        let i2 = indices[index_offset + 2];

        let v0 = detail::unpack_triangle_vertex(vertices[vertex_offset + i0]);
        let v1 = detail::unpack_triangle_vertex(vertices[vertex_offset + i1]);
        let v2 = detail::unpack_triangle_vertex(vertices[vertex_offset + i2]);

        // barycentric weights
        let bw = float3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x, hit.barycentrics.y);

        let pos_os = bw.x * v0.position + bw.y * v1.position + bw.z * v2.position;
        let shading_normal_os = bw.x * v0.normal + bw.y * v1.normal + bw.z * v2.normal;
        let shading_tangent_os = bw.x * v0.tangent + bw.y * v1.tangent + bw.z * v2.tangent;
        let uv = bw.x * v0.uv[0] + bw.y * v1.uv[0] + bw.z * v2.uv[0];

        let is_front_face_cw = false;
        let N = normalize(cross(v1.position - v0.position, v2.position - v0.position));
        let normal_os = select(is_front_face_cw, -N, N);
        let tangent_os = normalize(v1.position - v0.position);

        let world_from_object = get_world_from_object(geometry_instance.header.transform_id);
        let object_from_world = get_object_from_world(geometry_instance.header.transform_id);
        let inverse_transpose = transpose(float3x3(object_from_world));
        // All handedness values should be the same for the three vertices of a triangle, so we can just use v0's here.
        let transform_handedness = determinant(float3x3(world_from_object)) < 0.0 ? -1.0 : 1.0;
        let shading_handedness = v0.handedness * transform_handedness;

        SurfaceInteraction si = {};
        si.wi_ws = wi_ws;
        si.position_ws = transform_point(world_from_object, pos_os);
        si.normal_ws = normalize(transform_vector(inverse_transpose, normal_os));
        if (ENABLE_PRECISE_RAY_OFFSETS) {
            let spawn = compute_precise_ray_spawn(
                v0.position,
                v1.position,
                v2.position,
                hit.barycentrics,
                world_from_object,
                object_from_world
            );
            if (spawn.offset > 0.0) {
                si.position_ws = spawn.position;
                si.normal_ws = select(is_front_face_cw, -spawn.normal, spawn.normal);
                si.ray_offset.set(spawn.offset);
            }
        }
        si.front_facing = dot(wi_ws, si.normal_ws) >= 0;
        si.uv = uv;
        si.shading_frame_ws = Frame(
            normalize(transform_vector(inverse_transpose, shading_normal_os)),
            normalize(transform_vector(world_from_object, shading_tangent_os)),
            shading_handedness
        );
        si.set_hit_info(hit);
        return si;
    }
}

public struct DeferredTriangleHitInfo : IDeferredHitInfo
{
    public float2 barycentrics;
    public IHitInfo create()
    {
        return TriangleHitInfo(GeometryInstanceID(InstanceID() + GeometryIndex2()), PrimitiveIndex(), barycentrics);
    }
};
