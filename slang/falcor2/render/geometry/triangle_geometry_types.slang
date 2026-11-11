// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/shared_types_compat.slangh"

#ifndef FALCOR_SHARED_HOST
implementing render;
#endif

#define UV_COUNT 2

public struct TriangleVertex
{
    public float3 position;
    public float3 normal;
    public float3 tangent;
    public float handedness;
    public float3 color;
    public float2 uv[UV_COUNT];
};

public struct PackedTriangleVertex
{
    // Versioning for future compatibility.
    public FALCOR_SHARED_CONSTANT uint VERSION = 2;

    public float3 position;
    public uint3 normal_and_tangent;
    public float2 uv;
};

namespace detail {
static float3 unpack_vertex_normal(const uint3 normal_and_tangent)
{
    return f16tof32(
        uint3(normal_and_tangent[0] & 0xffff, (normal_and_tangent[0] >> 16) & 0xffff, normal_and_tangent[1] & 0xffff)
    );
}

static float unpack_vertex_handedness(const uint3 normal_and_tangent)
{
    return f16tof32((normal_and_tangent[1] >> 16) & 0xffff);
}

static float3 unpack_vertex_tangent(const uint3 normal_and_tangent)
{
    return unpack_normal_oct_snorm2x16(normal_and_tangent[2]);
}

public static void pack_vertex_normal(FALCOR_SHARED_INOUT_PARAM(uint3, normal_and_tangent), float3 normal)
{
    uint3 packed_normal = f32tof16(normal);
    normal_and_tangent[0] = (packed_normal.x & 0xffff) | ((packed_normal.y & 0xffff) << 16);
    normal_and_tangent[1] = (normal_and_tangent[1] & 0xffff0000) | (packed_normal.z & 0xffff);
}

static void pack_vertex_handedness(FALCOR_SHARED_INOUT_PARAM(uint3, normal_and_tangent), float handedness)
{
    uint packed_handedness = f32tof16(handedness);
    normal_and_tangent[1] = (normal_and_tangent[1] & 0x0000ffff) | ((packed_handedness & 0xffff) << 16);
}

public static void pack_vertex_tangent(FALCOR_SHARED_INOUT_PARAM(uint3, normal_and_tangent), float3 tangent)
{
    normal_and_tangent[2] = pack_normal_oct_snorm2x16(tangent);
}

static PackedTriangleVertex pack_triangle_vertex(const TriangleVertex v)
{
    PackedTriangleVertex pv;
    pv.position[0] = v.position.x;
    pv.position[1] = v.position.y;
    pv.position[2] = v.position.z;
    pv.normal_and_tangent[0] = 0;
    pv.normal_and_tangent[1] = 0;
    pv.normal_and_tangent[2] = 0;
    pack_vertex_normal(pv.normal_and_tangent, v.normal);
    pack_vertex_handedness(pv.normal_and_tangent, v.handedness);
    pack_vertex_tangent(pv.normal_and_tangent, v.tangent);
    pv.uv[0] = v.uv[0].x;
    pv.uv[1] = v.uv[0].y;
    return pv;
}
public static TriangleVertex unpack_triangle_vertex(const PackedTriangleVertex pv)
{
    TriangleVertex v = {};
    v.position = float3(pv.position[0], pv.position[1], pv.position[2]);
    v.normal = unpack_vertex_normal(pv.normal_and_tangent);
    v.handedness = unpack_vertex_handedness(pv.normal_and_tangent);
    v.tangent = unpack_vertex_tangent(pv.normal_and_tangent);
    v.uv[0] = float2(pv.uv[0], pv.uv[1]);
    return v;
}
} // namespace detail

public struct TriangleGeometryDesc
{
    public uint16_t vertex_buffer; ///< Vertex buffer slot.
    public uint16_t index_buffer;  ///< Index buffer slot.

    public uint vertex_offset; ///< Offset into vertex buffer.
    public uint index_offset;  ///< Offset into index buffer.
    public uint index_count;   ///< Index count; non-indexed geometry is not supported.
};

// Note: until we're certain of types to support for custom streams,
// keeping this as a separate enum to the SGL DataType / Format.
public enum class AdditionalAttributeType : uint {
    invalid = 0,
    bool_ = 1,   ///< Boolean value stored as a normalized 8-bit unsigned integer.
    int32 = 2,   ///< Signed 32-bit integer.
    float16 = 3, ///< IEEE 16-bit floating-point value.
    float32 = 4, ///< IEEE 32-bit floating-point value.
};

public struct PackedAdditionalAttribute
{
    public uint byte_offset;
    public uint format;
};

public static const uint MAX_ADDITIONAL_GEOMETRY_ATTRIBUTES = 32;
public static const uint ADDITIONAL_ATTRIBUTE_DESC_ROW_SIZE = MAX_ADDITIONAL_GEOMETRY_ATTRIBUTES + 1;

public struct TriangleGeometryInstanceDesc
{
    public GeometryInstanceDescHeader header;
    public uint16_t vertex_buffer; ///< Vertex buffer slot.
    public uint16_t index_buffer;  ///< Index buffer slot.

    public uint vertex_offset; ///< Offset into vertex buffer.
    public uint index_offset;  ///< Offset into index buffer, or zero if non-indexed.
    // Preserve the common GeometryInstanceDesc payload size for reinterpretation.
    public uint reserved;
};

#include "falcor2/shared_types_compat_undef.slangh"
