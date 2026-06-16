// SPDX-License-Identifier: Apache-2.0

#ifndef FALCOR_SHARED_HOST
implementing render;
#endif

// ----------------------------------------------------------------------------
// Emissive triangle types
// ----------------------------------------------------------------------------

/// Emissive triangle ID.
public enum class EmissiveTriangleID : uint { invalid = 0xffffffff };

/// Active emissive triangle ID.
public enum class ActiveEmissiveTriangleID : uint { invalid = 0xffffffff };

/// Emissive triangle.
/// Represents a single emissive triangle in world-space.
public struct EmissiveTriangle
{
    /// Position in world-space.
    public float3 pos_ws[3];
    /// Texture coordinates.
    public float2 uv[3];
    /// Normal in world-space.
    public float3 normal_ws;
    /// Area in world-space.
    public float area_ws;
    /// Geometry instance this triangle belongs to.
    public GeometryInstanceID geometry_instance_id;
    /// Material this triangle belongs to.
    public MaterialID material_id;

    /// Returns the interpolated position.
    /// @param bary Barycentric coordinates.
    /// @return Interpolated position in world space.
    public float3 get_pos_ws(float3 bary) { return pos_ws[0] * bary[0] + pos_ws[1] * bary[1] + pos_ws[2] * bary[2]; }

    /// Returns the interpolated texture coordinate.
    /// @param bary Barycentric coordinates.
    /// @return Interpolated texture coordinate.
    public float2 get_uv(float3 bary) { return uv[0] * bary[0] + uv[1] * bary[1] + uv[2] * bary[2]; }
};

/// Packed emissive triangle.
/// Used to store emissive triangles in buffers.
public struct PackedEmissiveTriangle
{
    /// Position in world-space.
    public float pos_ws[3][3];
    /// Texture coordinates (2x fp16 each).
    public uint uv[3];
    /// Normal in world-space (16-bit snorms, octahedral mapping).
    uint normal_ws;
    /// Area in world-space.
    float area_ws;
    /// Geometry instance this triangle belongs to.
    GeometryInstanceID geometry_instance_id;
    /// Material this triangle belongs to.
    MaterialID material_id;
};

namespace detail {

/// Pack an emissive triangle.
public static PackedEmissiveTriangle pack_emissive_triangle(EmissiveTriangle t)
{
    PackedEmissiveTriangle pt;
#ifndef FALCOR_SHARED_HOST
    [ForceUnroll]
#endif
    for (uint i = 0; i < 3; i++) {
        pt.pos_ws[i][0] = t.pos_ws[i].x;
        pt.pos_ws[i][1] = t.pos_ws[i].y;
        pt.pos_ws[i][2] = t.pos_ws[i].z;
        uint2 uv16 = f32tof16(t.uv[i]);
        pt.uv[i] = (uv16.x << 16) | uv16.y;
    }
    pt.normal_ws = pack_normal_oct_snorm2x16(t.normal_ws);
    pt.area_ws = t.area_ws;
    pt.geometry_instance_id = t.geometry_instance_id;
    pt.material_id = t.material_id;
    return pt;
}

/// Unpack an emissive triangle.
public static EmissiveTriangle unpack_emissive_triangle(PackedEmissiveTriangle pt)
{
    EmissiveTriangle t;
#ifndef FALCOR_SHARED_HOST
    [ForceUnroll]
#endif
    for (uint i = 0; i < 3; i++) {
        t.pos_ws[i] = float3(pt.pos_ws[i][0], pt.pos_ws[i][1], pt.pos_ws[i][2]);
        uint2 uv16 = uint2((pt.uv[i] >> 16) & 0xffff, pt.uv[i] & 0xffff);
        t.uv[i] = f16tof32(uv16);
    }
    t.normal_ws = unpack_normal_oct_snorm2x16(pt.normal_ws);
    t.area_ws = pt.area_ws;
    t.geometry_instance_id = pt.geometry_instance_id;
    t.material_id = pt.material_id;
    return t;
}

} // namespace detail
