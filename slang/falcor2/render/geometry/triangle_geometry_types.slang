// SPDX-License-Identifier: Apache-2.0

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
    public static constexpr uint VERSION = 2;

    public float position[3];
    public uint normal_and_tangent[3];
    public float uv[2];
};

#ifdef FALCOR_SHARED_HOST
#define NT_PARAM(name) uint(&name) [3]
#define NT_CONST_PARAM(name) const uint(&name) [3]
#else
#define NT_PARAM(name) inout uint name[3]
#define NT_CONST_PARAM(name) const uint name[3]
#endif

namespace detail {
static float3 unpack_vertex_normal(NT_CONST_PARAM(normal_and_tangent))
{
    return f16tof32(
        uint3(normal_and_tangent[0] & 0xffff, (normal_and_tangent[0] >> 16) & 0xffff, normal_and_tangent[1] & 0xffff)
    );
}

static float unpack_vertex_handedness(NT_CONST_PARAM(normal_and_tangent))
{
    return f16tof32((normal_and_tangent[1] >> 16) & 0xffff);
}

static float3 unpack_vertex_tangent(NT_CONST_PARAM(normal_and_tangent))
{
    return unpack_normal_oct_snorm2x16(normal_and_tangent[2]);
}

static void pack_vertex_normal(NT_PARAM(normal_and_tangent), float3 normal)
{
    uint3 packed_normal = f32tof16(normal);
    normal_and_tangent[0] = (packed_normal.x & 0xffff) | ((packed_normal.y & 0xffff) << 16);
    normal_and_tangent[1] = (normal_and_tangent[1] & 0xffff0000) | (packed_normal.z & 0xffff);
}

static void pack_vertex_handedness(NT_PARAM(normal_and_tangent), float handedness)
{
    uint packed_handedness = f32tof16(handedness);
    normal_and_tangent[1] = (normal_and_tangent[1] & 0x0000ffff) | ((packed_handedness & 0xffff) << 16);
}

static void pack_vertex_tangent(NT_PARAM(normal_and_tangent), float3 tangent)
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
static TriangleVertex unpack_triangle_vertex(const PackedTriangleVertex pv)
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

#undef NT_CONST_PARAM
#undef NT_PARAM

public struct TriangleGeometryDesc
{
    public uint vertex_offset; ///< Offset into vertex buffer.
    public uint vertex_count;  ///< Vertex count.
    public uint index_offset;  ///< Offset into index buffer, or zero if non-indexed.
    public uint index_count;   ///< Index count, or zero if non-indexed.
};

public struct TriangleGeometryInstanceDesc
{
    public GeometryInstanceDescHeader header;
    public uint vertex_offset; ///< Offset into vertex buffer.
    public uint vertex_count;  ///< Vertex count.
    public uint index_offset;  ///< Offset into index buffer, or zero if non-indexed.
    public uint index_count;   ///< Index count, or zero if non-indexed.
};
