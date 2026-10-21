// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/shared_types_compat.slangh"

#ifndef FALCOR_SHARED_HOST
implementing render;
#endif

public struct LSSVertex
{
    public float3 position;
    public float radius;
    public float2 texcoord;
};

public struct PackedLSSVertex
{
    // Versioning for future compatibility.
    public FALCOR_SHARED_CONSTANT uint VERSION = 2;

    public float position[3];
    public float radius;
    public float texcoord[2];
};

namespace detail {
public static PackedLSSVertex pack_lss_vertex(const LSSVertex v)
{
    PackedLSSVertex pv;
    pv.position[0] = v.position.x;
    pv.position[1] = v.position.y;
    pv.position[2] = v.position.z;
    pv.radius = v.radius;
    pv.texcoord[0] = v.texcoord.x;
    pv.texcoord[1] = v.texcoord.y;
    return pv;
}
public static LSSVertex unpack_lss_vertex(const PackedLSSVertex pv)
{
    LSSVertex v;
    v.position = float3(pv.position[0], pv.position[1], pv.position[2]);
    v.radius = pv.radius;
    v.texcoord = float2(pv.texcoord[0], pv.texcoord[1]);
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

#include "falcor2/shared_types_compat_undef.slangh"
