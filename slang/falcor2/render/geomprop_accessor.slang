// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

static const uint GEOMETRY_ATTRIBUTE_ID_COLOR_0 = 8;

public enum class GeomPropSpace : uint {
    model = 0,
    object = 1,
    world = 2,
};

static const float4x4 GEOMETRY_IDENTITY_4X4 = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

public float4x4 geomprop_identity4x4()
{
    return GEOMETRY_IDENTITY_4X4;
}

/// Geometry properties evaluated at one surface interaction.
public interface IGeomPropAccessor
{
    /// Descriptor presence, independent of the reader's value type or interpolation mode.
    public bool has_attribute(uint id);
    public bool try_get(uint id, out int value);
    public bool try_get(uint id, out int2 value);
    public bool try_get(uint id, out int3 value);
    public bool try_get(uint id, out int4 value);
    public bool try_get(uint id, out bool value);
    public bool try_get(uint id, out bool2 value);
    public bool try_get(uint id, out bool3 value);
    public bool try_get(uint id, out bool4 value);
    public bool try_get(uint id, out float value);
    public bool try_get(uint id, out float2 value);
    public bool try_get(uint id, out float3 value);
    public bool try_get(uint id, out float4 value);

    public int get_int(uint id, int fallback = 0)
    {
        int value;
        return try_get(id, value) ? value : fallback;
    }

    public int2 get_int2(uint id, int2 fallback = 0)
    {
        int2 value;
        return try_get(id, value) ? value : fallback;
    }

    public int3 get_int3(uint id, int3 fallback = 0)
    {
        int3 value;
        return try_get(id, value) ? value : fallback;
    }

    public int4 get_int4(uint id, int4 fallback = 0)
    {
        int4 value;
        return try_get(id, value) ? value : fallback;
    }

    public bool get_bool(uint id, bool fallback = false)
    {
        bool value;
        return try_get(id, value) ? value : fallback;
    }

    public bool2 get_bool2(uint id, bool2 fallback = false)
    {
        bool2 value;
        return try_get(id, value) ? value : fallback;
    }

    public bool3 get_bool3(uint id, bool3 fallback = false)
    {
        bool3 value;
        return try_get(id, value) ? value : fallback;
    }

    public bool4 get_bool4(uint id, bool4 fallback = false)
    {
        bool4 value;
        return try_get(id, value) ? value : fallback;
    }

    public float get_float(uint id, float fallback = 0.f)
    {
        float value;
        return try_get(id, value) ? value : fallback;
    }

    public float2 get_float2(uint id, float2 fallback = 0.f)
    {
        float2 value;
        return try_get(id, value) ? value : fallback;
    }

    public float3 get_float3(uint id, float3 fallback = 0.f)
    {
        float3 value;
        return try_get(id, value) ? value : fallback;
    }

    public float4 get_float4(uint id, float4 fallback = 0.f)
    {
        float4 value;
        return try_get(id, value) ? value : fallback;
    }

    public bool try_get_additional_texcoord(uint index, out float2 value);
    public bool try_get_additional_texcoord(uint index, out float3 value);
    public bool try_get_additional_texcoord(uint index, out float4 value);
    public bool try_get_color(uint index, out float4 value);

    public float2 get_additional_texcoord(uint index, float2 fallback)
    {
        float2 value;
        return try_get_additional_texcoord(index, value) ? value : fallback;
    }

    public float3 get_additional_texcoord(uint index, float3 fallback)
    {
        float3 value;
        return try_get_additional_texcoord(index, value) ? value : fallback;
    }

    public float4 get_additional_texcoord(uint index, float4 fallback)
    {
        float4 value;
        return try_get_additional_texcoord(index, value) ? value : fallback;
    }

    public float4 get_color(uint index, float4 fallback = 0.f)
    {
        float4 value;
        return try_get_color(index, value) ? value : fallback;
    }

    public float4x4 get_space_transform(GeomPropSpace to_space, GeomPropSpace from_space);
};

public struct InvalidGeomPropAccessor : IGeomPropAccessor
{
    public bool has_attribute(uint id) { return false; }
    public bool try_get(uint id, out int value)
    {
        value = 0;
        return false;
    }

    public bool try_get(uint id, out int2 value)
    {
        value = 0;
        return false;
    }

    public bool try_get(uint id, out int3 value)
    {
        value = 0;
        return false;
    }

    public bool try_get(uint id, out int4 value)
    {
        value = 0;
        return false;
    }

    public bool try_get(uint id, out bool value)
    {
        value = false;
        return false;
    }

    public bool try_get(uint id, out bool2 value)
    {
        value = false;
        return false;
    }

    public bool try_get(uint id, out bool3 value)
    {
        value = false;
        return false;
    }

    public bool try_get(uint id, out bool4 value)
    {
        value = false;
        return false;
    }

    public bool try_get(uint id, out float value)
    {
        value = 0.f;
        return false;
    }

    public bool try_get(uint id, out float2 value)
    {
        value = 0.f;
        return false;
    }

    public bool try_get(uint id, out float3 value)
    {
        value = 0.f;
        return false;
    }

    public bool try_get(uint id, out float4 value)
    {
        value = 0.f;
        return false;
    }

    public bool try_get_additional_texcoord(uint index, out float2 value)
    {
        value = 0.f;
        return false;
    }

    public bool try_get_additional_texcoord(uint index, out float3 value)
    {
        value = 0.f;
        return false;
    }

    public bool try_get_additional_texcoord(uint index, out float4 value)
    {
        value = 0.f;
        return false;
    }

    public bool try_get_color(uint index, out float4 value)
    {
        value = 0.f;
        return false;
    }

    public float4x4 get_space_transform(GeomPropSpace to_space, GeomPropSpace from_space)
    {
        return geomprop_identity4x4();
    }
};

// clang-format off
/// Acquire the production accessor from retained hit context without retaining a full SurfaceInteraction.
// WAR: Inline before type-flow specialization; ordinary ForceInline runs too late for combined MDL reads.
// Remove when https://github.com/shader-slang/slang/issues/13046 is fixed in the pinned compiler.
[__unsafeForceInlineEarly]
public IGeomPropAccessor get_hit_geomprop_accessor(Conditional<PackedHitInfo, ENABLE_GEOMETRY_PROPERTIES> hit_info)
{
    if (let packed_hit = hit_info.get()) {
        // TODO: Avoid nesting an unpacked IHitInfo existential inside the returned accessor existential.
        // This triggers a Slang autodiff compiler failure when scene_test composes its path-tracing and debug ray generation.
        // https://github.com/shader-slang/slang/issues/12934
        // const IHitInfo hit = packed_hit.unpack();
        // if (let triangle_hit = hit as TriangleHitInfo)
        //     return TriangleGeomPropAccessor(triangle_hit);
        // if (let lss_hit = hit as LSSHitInfo)
        //     return LSSGeomPropAccessor(lss_hit);
        if (packed_hit.geometry_type == GeometryType::triangle)
            return TriangleGeomPropAccessor(TriangleHitInfo(packed_hit));
        if (packed_hit.geometry_type == GeometryType::lss)
            return LSSGeomPropAccessor(LSSHitInfo(packed_hit));
    }
    return InvalidGeomPropAccessor();
}

public extension SurfaceInteraction
{
    public IGeomPropAccessor get_geomprop_accessor()
    {
        return get_hit_geomprop_accessor(hit_info);
    }
};
// clang-format on
