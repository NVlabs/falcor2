// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;

public struct LSSGeometryData
{
    public StructuredBuffer<LSSGeometryDesc> descs;
    public StructuredBuffer<PackedAdditionalAttribute> additional_attributes;
    public ByteAddressBuffer additional_attribute_data;
};

namespace detail {
float2x4 load_lss_segment<let HasTracingIntrinsics : bool = false>(
    StructuredBuffer<BufferPtr> geometry,
    uint buffer_index,
    uint first_vertex
)
{
    if (HasTracingIntrinsics) {
        // Stage selection keeps the default specialization callable from compute/query shaders.
        __stage_switch
        {
        case closesthit:
        case anyhit:
            return GetLssPositionsAndRadii();
        default:
            break;
        }
    }

    let vertices = geometry[buffer_index].as<PackedLSSVertex>();
    const LSSVertex v0 = detail::unpack_lss_vertex(vertices[first_vertex]);
    const LSSVertex v1 = detail::unpack_lss_vertex(vertices[first_vertex + 1]);
    return float2x4(float4(v0.position, v0.radius), float4(v1.position, v1.radius));
}
}

public extension Scene
{
    public LSSVertex get_lss_vertex(uint buffer_index, uint vertex_index)
    {
        return detail::unpack_lss_vertex(geometry[buffer_index].as<PackedLSSVertex>()[vertex_index]);
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

        let i0 = geometry[instance_desc.index_buffer].as<uint>()[index_offset];
        return detail::load_lss_segment(geometry, instance_desc.vertex_buffer, vertex_offset + i0);
    }

    /// Previous endpoints and radii occupy vertex_buffer + 1 with the current topology and vertex offsets.
    /// Requires vertex_motion and a two-buffer vertex tuple with previous-frame data populated.
    public float3 get_previous_position_os(LSSHitInfo hit)
    {
        LSSGeometryInstanceDesc instance_desc
            = reinterpret<LSSGeometryInstanceDesc>(get_geometry_instance_desc(hit.geometry_instance_id));
        let index_offset = instance_desc.index_offset + hit.primitive_index;
        let i0 = geometry[instance_desc.index_buffer].as<uint>()[index_offset];
        let segment
            = detail::load_lss_segment(geometry, instance_desc.vertex_buffer + 1, instance_desc.vertex_offset + i0);
        float3 position_os;
        float3 sphere_center;
        curve_parameters_to_hit_pos(hit.curve_parameters, segment[0], segment[1], position_os, sphere_center);
        return position_os;
    }
}

public struct LSSHitInfo : IHitInfo
{
    public property GeometryType geometry_type
    {
        get {
            return GeometryType::lss;
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

    /// Cap-inclusive axial position and angle around the segment, both normalized to [0, 1].
    /// Coincident endpoints use +Z and the larger radius as a best-effort fallback;
    /// UVs and attributes use the resulting axial weight, without continuity guarantees.
    public float2 curve_parameters;

    public __init(
        GeometryInstanceID geometry_instance_id,
        uint primitive_index,
        float3 hit_pos_os,
        float2x4 positions_and_radii_os
    )
    {
        this.geometry_instance_id = geometry_instance_id;
        this.primitive_index = primitive_index;
        this.curve_parameters
            = curve_hit_pos_to_parameters(hit_pos_os, positions_and_radii_os[0], positions_and_radii_os[1]);
    }

    public __init(const PackedHitInfo p)
    {
        geometry_instance_id = p.geometry_instance_id;
        primitive_index = p.payload[0];
        curve_parameters = float2(asfloat(p.payload[1]), asfloat(p.payload[2]));
    }

    public PackedHitInfo pack()
    {
        PackedHitInfo p = {};
        p.geometry_type = geometry_type;
        p.geometry_instance_id = geometry_instance_id;
        p.payload[0] = primitive_index;
        p.payload[1] = asuint(curve_parameters.x);
        p.payload[2] = asuint(curve_parameters.y);
        return p;
    }
};

public struct LSSGeomPropAccessor : IGeomPropAccessor
{
    public no_diff GeometryInstanceID geometry_instance_id;
    public no_diff uint additional_attribute_desc_offset;
    public no_diff uint additional_attribute_mask;
    public no_diff uint2 local_vertex_indices;
    public no_diff float2 endpoint_weights;

    public __init(const LSSHitInfo hit)
    {
        geometry_instance_id = hit.geometry_instance_id;
        const GeometryInstanceDesc geometry_instance = g_scene.get_geometry_instance_desc(geometry_instance_id);
        const LSSGeometryInstanceDesc lss_geometry_instance = reinterpret<LSSGeometryInstanceDesc>(geometry_instance);
        additional_attribute_desc_offset
            = uint(lss_geometry_instance.header.geometry_id) * ADDITIONAL_ATTRIBUTE_DESC_ROW_SIZE;
        additional_attribute_mask
            = g_scene.lss_geometry.additional_attributes[additional_attribute_desc_offset].byte_offset;
        const uint index_offset = lss_geometry_instance.index_offset + hit.primitive_index;
        const uint first_vertex_index = g_scene.geometry[lss_geometry_instance.index_buffer].as<uint>()[index_offset];
        local_vertex_indices = uint2(first_vertex_index, first_vertex_index + 1);
        endpoint_weights = float2(1.f - hit.curve_parameters.x, hit.curve_parameters.x);
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
        attribute = g_scene.lss_geometry.additional_attributes[additional_attribute_desc_offset + 1 + attribute_rank];
        component_count = attribute.format & 0xff;
        type = AdditionalAttributeType(attribute.format >> 8);
        return true;
    }

    uint get_dominant_local_vertex_index()
    {
        return endpoint_weights.y > endpoint_weights.x ? local_vertex_indices.y : local_vertex_indices.x;
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

        float4 loaded_values[2];
        [ForceUnroll]
        for (uint vertex = 0; vertex < 2; ++vertex)
            loaded_values[vertex] = 0.f;

        if (type == AdditionalAttributeType::float16) {
            const uint2 addresses = attribute.byte_offset + local_vertex_indices * component_count * sizeof(float16_t);
            switch (component_count) {
            case 1:
                [ForceUnroll] for (uint vertex = 0; vertex < 2; ++vertex) loaded_values[vertex].x
                    = float(g_scene.lss_geometry.additional_attribute_data.LoadAligned<float16_t>(addresses[vertex]));
                break;
            case 2:
                [ForceUnroll] for (uint vertex = 0; vertex < 2; ++vertex) loaded_values[vertex].xy
                    = float2(g_scene.lss_geometry.additional_attribute_data.LoadAligned<float16_t2>(addresses[vertex]));
                break;
            case 3:
                [ForceUnroll] for (uint vertex = 0; vertex < 2; ++vertex) loaded_values[vertex].xyz
                    = float3(g_scene.lss_geometry.additional_attribute_data.LoadAligned<float16_t3>(addresses[vertex]));
                break;
            case 4:
                [ForceUnroll] for (uint vertex = 0; vertex < 2; ++vertex) loaded_values[vertex]
                    = float4(g_scene.lss_geometry.additional_attribute_data.LoadAligned<float16_t4>(addresses[vertex]));
                break;
            }
        } else {
            const uint2 addresses = attribute.byte_offset + local_vertex_indices * component_count * sizeof(float);
            switch (component_count) {
            case 1:
                [ForceUnroll] for (uint vertex = 0; vertex < 2; ++vertex) loaded_values[vertex].x
                    = g_scene.lss_geometry.additional_attribute_data.LoadAligned<float>(addresses[vertex]);
                break;
            case 2:
                [ForceUnroll] for (uint vertex = 0; vertex < 2; ++vertex) loaded_values[vertex].xy
                    = g_scene.lss_geometry.additional_attribute_data.LoadAligned<float2>(addresses[vertex]);
                break;
            case 3:
                [ForceUnroll] for (uint vertex = 0; vertex < 2; ++vertex) loaded_values[vertex].xyz
                    = g_scene.lss_geometry.additional_attribute_data.LoadAligned<float3>(addresses[vertex]);
                break;
            case 4:
                [ForceUnroll] for (uint vertex = 0; vertex < 2; ++vertex) loaded_values[vertex]
                    = g_scene.lss_geometry.additional_attribute_data.LoadAligned<float4>(addresses[vertex]);
                break;
            }
        }

        const float4 interpolated_value = endpoint_weights.x * loaded_values[0] + endpoint_weights.y * loaded_values[1];
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
            loaded_value.x = g_scene.lss_geometry.additional_attribute_data.LoadAligned<int>(address);
            break;
        case 2:
            loaded_value.xy = g_scene.lss_geometry.additional_attribute_data.LoadAligned<int2>(address);
            break;
        case 3:
            loaded_value.xyz = g_scene.lss_geometry.additional_attribute_data.LoadAligned<int3>(address);
            break;
        case 4:
            loaded_value = g_scene.lss_geometry.additional_attribute_data.LoadAligned<int4>(address);
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
        uint value
            = g_scene.lss_geometry.additional_attribute_data.LoadAligned<uint>(aligned_address) >> (byte_offset * 8u);
        if (byte_offset + uint(ComponentCount) > 4u) {
            value |= g_scene.lss_geometry.additional_attribute_data.LoadAligned<uint>(aligned_address + 4u)
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
            const uint packed_value = g_scene.lss_geometry.additional_attribute_data.LoadAligned<uint>(address);
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

    public bool try_get(uint id, out int value)
    {
        value = 0;
        vector<int, 1> loaded_value;
        if (!try_load_int_attribute(id, loaded_value))
            return false;
        value = loaded_value.x;
        return true;
    }

    public bool try_get(uint id, out int2 value) { return try_load_int_attribute(id, value); }

    public bool try_get(uint id, out int3 value) { return try_load_int_attribute(id, value); }

    public bool try_get(uint id, out int4 value) { return try_load_int_attribute(id, value); }

    public bool try_get(uint id, out bool value)
    {
        value = false;
        vector<bool, 1> loaded_value;
        if (!try_load_bool_attribute(id, loaded_value))
            return false;
        value = loaded_value.x;
        return true;
    }

    public bool try_get(uint id, out bool2 value) { return try_load_bool_attribute(id, value); }

    public bool try_get(uint id, out bool3 value) { return try_load_bool_attribute(id, value); }

    public bool try_get(uint id, out bool4 value) { return try_load_bool_attribute(id, value); }

    public bool try_get(uint id, out float value)
    {
        value = 0.f;
        vector<float, 1> loaded_value;
        if (!try_load_attribute(id, loaded_value))
            return false;
        value = loaded_value.x;
        return true;
    }

    public bool try_get(uint id, out float2 value) { return try_load_attribute(id, value); }

    public bool try_get(uint id, out float3 value) { return try_load_attribute(id, value); }

    public bool try_get(uint id, out float4 value) { return try_load_attribute(id, value); }

    public bool try_get_additional_texcoord(uint index, out float2 value)
    {
        value = 0.f;
        return index < 8 && try_get(index, value);
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
    /// Construct a surface interaction from an LSS hit.
    /// Enable only for the current native LSS hit in a ray-tracing hit shader.
    public SurfaceInteraction make_lss_surface_interaction<let HasTracingIntrinsics : bool = false>(
        LSSHitInfo hit,
        float3 wi_ws
    )
    {
        GeometryInstanceDesc geometry_instance = get_geometry_instance_desc(hit.geometry_instance_id);
        LSSGeometryInstanceDesc geometry_instance_data = reinterpret<LSSGeometryInstanceDesc>(geometry_instance);

        let vertex_offset = geometry_instance_data.vertex_offset;
        let index_offset = geometry_instance_data.index_offset + hit.primitive_index;

        let vertices = geometry[geometry_instance_data.vertex_buffer].as<PackedLSSVertex>();
        let i0 = geometry[geometry_instance_data.index_buffer].as<uint>()[index_offset];
        let first_vertex = vertex_offset + i0;
        let endpoints = detail::load_lss_segment<HasTracingIntrinsics>(
            geometry,
            geometry_instance_data.vertex_buffer,
            first_vertex
        );
        let v0 = LSSVertex(endpoints[0].xyz, endpoints[0].w, vertices[first_vertex].texcoord);
        let v1 = LSSVertex(endpoints[1].xyz, endpoints[1].w, vertices[first_vertex + 1].texcoord);

        float3 hit_pos;
        float3 sphere_center;
        curve_parameters_to_hit_pos(
            hit.curve_parameters,
            float4(v0.position, v0.radius),
            float4(v1.position, v1.radius),
            hit_pos,
            sphere_center
        );

        let pos_os = hit_pos;
        let normal_os = hit_pos - sphere_center;
        let tangent_os = v1.position - v0.position;

        let world_from_object = get_world_from_object(geometry_instance.header.transform_id);
        let inverse_transpose = transpose(float3x3(get_object_from_world(geometry_instance.header.transform_id)));

        SurfaceInteraction si = {};
        si.wi_ws = wi_ws;
        si.position_ws = transform_point(world_from_object, pos_os);
        si.normal_ws = normalize(transform_vector(inverse_transpose, normal_os));
        si.front_facing = dot(wi_ws, si.normal_ws) >= 0;
        si.uv = lerp(v0.texcoord, v1.texcoord, hit.curve_parameters.x);
        // A collapsed segment has no tangent. Preserve the sphere normal, including at its poles.
        si.shading_frame_ws = all(v0.position == v1.position)
            ? Frame(si.normal_ws)
            : Frame(
                  si.normal_ws,
                  normalize(transform_vector(world_from_object, tangent_os)),
                  1.0 /*handedness*/,
                  Frame::Component::tangent
              );
        si.set_hit_info(hit);
        return si;
    }
}

/// Compute normalized axial position (including caps) and angle from a ray-curve hit.
///
/// @param hit_pos Position of the ray-curve hit point.
/// @param sphere_a Sphere (3D position + radius) at one end point.
/// @param sphere_b Sphere (3D position + radius) at the other end point.
/// @return Geometric curve parameters, both in [0, 1].
float2 curve_hit_pos_to_parameters(float3 hit_pos, float4 sphere_a, float4 sphere_b)
{
    // Build a reference frame.
    float3 fwd = sphere_b.xyz - sphere_a.xyz;
    float fwd_length = length(fwd);
    if (all(fwd == 0.0)) {
        sphere_a.w = sphere_b.w = max(sphere_a.w, sphere_b.w);
        fwd = float3(0, 0, 1);
    } else {
        fwd /= fwd_length;
    }
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
    result.y = x == 0.0 && y == 0.0 ? 0.5 : atan2(y, x) / float.TWO_PI + 0.5;
    return result;
}

/// Reconstruct the ray-curve hit position and sphere center from geometric curve parameters.
/// Uses Han et al. (2019), Section 3.2.2, for the tangent-cone geometry and clipping planes:
/// https://doi.org/10.1111/cgf.13703. The cap-inclusive axial/azimuthal coordinates are Falcor's parameterization.
///
/// @param curve_parameters Cap-inclusive axial position and angle of the hit point.
/// @param sphere_a Sphere (3D position + radius) at one end point.
/// @param sphere_b Sphere (3D position + radius) at the other end point.
/// @param[out] hit_pos Position of the ray-curve hit point.
/// @param[out] sphere_center Position of the corresponding sphere center (for computing the normal direction).
void curve_parameters_to_hit_pos(
    float2 curve_parameters,
    float4 sphere_a,
    float4 sphere_b,
    out float3 hit_pos,
    out float3 sphere_center
)
{
    // Build a reference frame.
    float3 fwd = sphere_b.xyz - sphere_a.xyz;
    float fwd_length = length(fwd);
    if (all(fwd == 0.0)) {
        sphere_a.w = sphere_b.w = max(sphere_a.w, sphere_b.w);
        fwd = float3(0, 0, 1);
    } else {
        fwd /= fwd_length;
    }
    float3 s, t;
    branchless_ONB(fwd, s, t);

    // Remapping.
    float z = curve_parameters.x * (sphere_a.w + sphere_b.w + fwd_length) - sphere_a.w;
    float phi = (curve_parameters.y - 0.5f) * float.TWO_PI;

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

/// Read the accepted TraceRay hit position in object space for the LSS deferred-hit adapters.
float3 get_lss_ray_tracing_hit_position()
{
#ifndef __TARGET_CUDA__
    float3 position_os = ObjectRayOrigin() + ObjectRayDirection() * RayTCurrent();
#else
    float3x4 world_to_object = WorldToObject3x4();
    float3 position_ws = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 position_os = transform_point(world_to_object, position_ws);
#endif
    return position_os;
}

public struct DeferredLSSHitInfo : IDeferredHitInfo
{
    public IHitInfo create()
    {
        float3 hit_pos_os = get_lss_ray_tracing_hit_position();
        return LSSHitInfo(
            GeometryInstanceID(InstanceID() + GeometryIndex2()),
            PrimitiveIndex(),
            hit_pos_os,
            GetLssPositionsAndRadii()
        );
    }
};

// ---------------------------------------------------------------------------
// Procedural LSS support
// ---------------------------------------------------------------------------

namespace detail {
// Entry root of A*t^2 + 2*B*t + C, i.e. the root with nonpositive derivative.
// This works for either sign of A and avoids subtracting nearly equal values.
bool lss_entry_root(float A, float B, float C, out float t)
{
    t = {};
    if (A == 0.0) {
        if (B >= 0.0)
            return false;
        t = -C / (2.0 * B);
        return true;
    }
    const float discriminant = B * B - A * C;
    if (discriminant < 0.0)
        return false;
    const float root = sqrt(discriminant);
    t = B < 0.0 ? C / (-B + root) : (-B - root) / A;
    return true;
}
}

/// Intersect the entry boundary of a swept sphere with nonnegative endpoint radii.
/// Directions need not be normalized. Caps select the exposed start/end sphere surfaces.
/// Inside origins and entries through a disabled cap produce no hit, including when t_min clips the entry.
/// Rejection is per segment; an enabled cap on a neighboring segment can still be hit from inside the strand.
/// Policy: https://docs.vulkan.org/spec/latest/chapters/raytraversal.html#ray-intersection-candidate-determination
public bool intersect_procedural_lss(
    const float3 ray_origin,
    const float3 ray_dir,
    const float t_min,
    const float t_max,
    const float2x4 segment,
    out float t,
    const bool2 caps = bool2(true)
)
{
    // Put a possible zero-radius tip at the equation origin. Swap its cap flag too.
    const bool reverse = segment[0].w > segment[1].w;
    const float2x4 spheres = reverse ? float2x4(segment[1], segment[0]) : segment;
    const bool2 enabled_caps = reverse ? caps.yx : caps;
    t = t_max;
    bool hit = false;
    const float dd = dot(ray_dir, ray_dir);
    if (dd == 0.0 || max(spheres[0].w, spheres[1].w) == 0.0)
        return false;

    const float3 axis = spheres[1].xyz - spheres[0].xyz;
    const float3 original_offset = ray_origin - spheres[0].xyz;
    // Center the equations near the segment for distant rays. Roots may be negative
    // in this coordinate system; choose the entry before restoring the original t.
    // Anchor at a zero-radius tip when present so its double root stays near zero.
    const float center_u = spheres[0].w == 0.0 ? 0.0 : 0.5;
    const float shift = dot(center_u * axis - original_offset, ray_dir) / dd;
    const float3 offset = original_offset + shift * ray_dir;
    const float dr = spheres[1].w - spheres[0].w;
    const float axis_length2 = dot(axis, axis);
    const float g = axis_length2 - dr * dr;
    const float y = dot(offset, axis) + spheres[0].w * dr;
    const float dy = dot(ray_dir, axis);

    // Direct endpoint quadratic; an equivalent derivation is given in Embree's ConeGeometryIntersector::intersectCone:
    // https://github.com/RenderKit/embree/blob/3d9cb89b9ea099c630e6272d37767e7dd4e78e74/kernels/geometry/roundline_intersector.h
    // Avoids the cone-apex singularity at equal radii. Cross products reduce cancellation for near-axis rays.
    // Coefficients use A*t^2 + 2*B*t + C, so Embree's B is twice ours.
    // y(t) / g is the interpolated sphere parameter; it also clips the body and caps.
    if (g > 0.0) {
        const float3 cross_origin = cross(axis, offset);
        const float3 cross_dir = cross(axis, ray_dir);
        const float A = dot(cross_dir, cross_dir) - dr * dr * dd;
        const float B = dot(cross_origin, cross_dir) - dr * (dr * dot(offset, ray_dir) + spheres[0].w * dy);
        // Keep the radius-difference term separate: reassociation loses it near
        // the tangent of a nearly cylindrical segment.
        const float C = fma(
            -dr,
            dr * dot(offset, offset) + 2.0 * spheres[0].w * dot(offset, axis),
            dot(cross_origin, cross_origin) - spheres[0].w * spheres[0].w * axis_length2
        );
        float candidate;
        // At a zero-radius tip the squared cone has a double root even for an
        // exit. Only the direction toward increasing radius enters the physical cone.
        const bool tip_exit = A < 0.0 && B * B == A * C && dr * dy <= 0.0;
        if (!tip_exit && detail::lss_entry_root(A, B, C, candidate)) {
            const float cy = y + candidate * dy;
            const float ray_t = candidate + shift;
            if (ray_t >= t_min && ray_t <= t && cy >= 0.0 && cy <= g) {
                t = ray_t;
                hit = true;
            }
        }
    }

    // A ray lying along a cylindrical body can touch both caps. Keep the nearer
    // cap candidate rather than returning the first endpoint sphere hit.
    [ForceUnroll]
    for (uint i = 0; i < 2; ++i) {
        if (!enabled_caps[i])
            continue;
        // A self-enclosing segment reduces to the larger sphere; coincident equal
        // spheres use whichever cap is enabled. A zero-radius endpoint has no surface.
        if (spheres[i].w == 0.0 || (g <= 0.0 && i != uint(dr > 0.0 || (dr == 0.0 && !enabled_caps.x))))
            continue;
        const float3 h = offset - float(i) * axis;
        // Evaluate the discriminant at closest approach to avoid cancellation
        // between the squared distance to the sphere and the ray projection.
        const float center_t = -dot(h, ray_dir) / dd;
        const float3 closest = h + center_t * ray_dir;
        const float discriminant = spheres[i].w * spheres[i].w - dot(closest, closest);
        if (discriminant >= 0.0) {
            const float candidate = center_t - sqrt(discriminant / dd);
            const float cy = dot(h + candidate * ray_dir, axis) + spheres[i].w * dr;
            const bool exposed = g <= 0.0 || (i == 0 ? cy <= 0.0 : cy >= 0.0);
            const float ray_t = candidate + shift;
            if (exposed && ray_t >= t_min && ray_t <= t) {
                t = ray_t;
                hit = true;
            }
        }
    }
    return hit;
}

public extension Scene
{
    /// Intersect an object-space ray using the scene's native chained-cap policy.
    public bool intersect_lss(
        GeometryInstanceID geometry_instance_id,
        uint primitive_index,
        float3 ray_origin,
        float3 ray_dir,
        float t_min,
        float t_max,
        out float t
    )
    {
        const LSSGeometryInstanceDesc desc
            = reinterpret<LSSGeometryInstanceDesc>(get_geometry_instance_desc(geometry_instance_id));
        const uint index = desc.index_offset + primitive_index;
        let indices = geometry[desc.index_buffer].as<uint>();
        const uint first_vertex = indices[index];
        // Match the successive indexing and chained end caps used by RenderScene:
        // VkRayTracingLssPrimitiveEndCapsModeNV enables both caps on a chain's first
        // primitive, and only the trailing cap on each following primitive.
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkRayTracingLssPrimitiveEndCapsModeNV.html
        bool start_cap = true;
        if (primitive_index > 0)
            start_cap = indices[index - 1] + 1 != first_vertex;
        return intersect_procedural_lss(
            ray_origin,
            ray_dir,
            t_min,
            t_max,
            detail::load_lss_segment(geometry, desc.vertex_buffer, desc.vertex_offset + first_vertex),
            t,
            bool2(start_cap, true)
        );
    }
}

public struct ProceduralLSSIntersectionAttributes
{
    float dummy = {};
};

public struct DeferredProceduralLSSHitInfo : IDeferredHitInfo
{
    public float2x4 segment;
    public IHitInfo create()
    {
        float3 hit_pos_os = get_lss_ray_tracing_hit_position();
        return LSSHitInfo(GeometryInstanceID(InstanceID() + GeometryIndex2()), PrimitiveIndex(), hit_pos_os, segment);
    }
};
