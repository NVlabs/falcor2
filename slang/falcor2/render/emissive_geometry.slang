// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct EmissiveGeometry
{
    /// Total number of (potentially) emissive triangles.
    public uint triangle_count;
    /// Total number of active emissive triangles that have non-zero emission.
    public uint active_triangle_count;

    /// Buffer containing (potentially) emissive triangles.
    public StructuredBuffer<PackedEmissiveTriangle> triangles;
    /// Buffer containing the emission values.
    public StructuredBuffer<float4> triangle_emission;
    /// Buffer containing the flux values.
    public StructuredBuffer<float> triangle_flux;

    /// Buffer containing mapping from GeometryInstanceID to EmissiveTriangleID.
    public StructuredBuffer<EmissiveTriangleID> geometry_instance_to_triangle_id;
    /// Buffer containing mapping from EmissiveTriangleID to ActiveEmissiveTriangleID.
    /// Contains ActiveEmissiveTriangle.invalid if triangle is non-emissive.
    public StructuredBuffer<ActiveEmissiveTriangleID> triangle_id_to_active_triangle_id;
    /// Buffer containing mapping from ActiveEmissiveTriangleID to EmissiveTriangleID.
    public StructuredBuffer<EmissiveTriangleID> active_triangle_id_to_triangle_id;

    public AliasTable1D active_triangle_alias_table;

    /// Returns true if there are active emissive triangles.
    public bool has_active_triangles() { return active_triangle_count != 0; }

    /// Get the EmissiveTriangleID given a geometry instance and primitive index.
    public EmissiveTriangleID get_triangle_id(GeometryInstanceID geometry_instance_id, uint primitive_index)
    {
        return EmissiveTriangleID(uint(geometry_instance_to_triangle_id[uint(geometry_instance_id)]) + primitive_index);
    }

    /// Get an EmissiveTriangle given an EmissiveTriangleID.
    public EmissiveTriangle get_triangle(EmissiveTriangleID id)
    {
        return detail::unpack_emissive_triangle(triangles[uint(id)]);
    }

    /// Get the triangle emission given an EmissiveTriangleID.
    public float3 get_triangle_emission(EmissiveTriangleID id) { return triangle_emission[uint(id)].rgb; }

    /// Get the triangle flux given an EmissiveTriangleID.
    public float get_triangle_flux(EmissiveTriangleID id) { return triangle_flux[uint(id)]; }
}
