// SPDX-License-Identifier: Apache-2.0

#ifndef FALCOR_SHARED_HOST
implementing render;
#endif

public struct LSSVertex
{
    public float3 position;
    public float radius;
};

public struct PackedLSSVertex
{
    // Versioning for future compatibility.
    public static constexpr uint VERSION = 1;

    public float position[3];
    public float radius;
};

namespace detail {
public static PackedLSSVertex pack_lss_vertex(const LSSVertex v)
{
    PackedLSSVertex pv;
    pv.position[0] = v.position.x;
    pv.position[1] = v.position.y;
    pv.position[2] = v.position.z;
    pv.radius = v.radius;
    return pv;
}
public static LSSVertex unpack_lss_vertex(const PackedLSSVertex pv)
{
    LSSVertex v;
    v.position = float3(pv.position[0], pv.position[1], pv.position[2]);
    v.radius = pv.radius;
    return v;
}
} // namespace detail

public struct LSSGeometryDesc
{
    public uint vertex_offset; ///< Offset into vertex buffer.
    public uint vertex_count;  ///< Vertex count.
    public uint index_offset;  ///< Offset into index buffer, or zero if non-indexed.
    public uint index_count;   ///< Index count, or zero if non-indexed.
};

public struct LSSGeometryInstanceDesc
{
    public GeometryInstanceDescHeader header;
    public uint vertex_offset; ///< Offset into vertex buffer.
    public uint vertex_count;  ///< Vertex count.
    public uint index_offset;  ///< Offset into index buffer, or zero if non-indexed.
    public uint index_count;   ///< Index count, or zero if non-indexed.
};
