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
    /// Buffer containing area-averaged RGB radiant exitance for each triangle.
    public StructuredBuffer<float4> triangle_emission;
    /// Buffer containing luminance-weighted one-sided emitted radiant flux for each triangle.
    public StructuredBuffer<float> triangle_flux;

    /// Buffer containing mapping from GeometryInstanceID to EmissiveTriangleID.
    public StructuredBuffer<EmissiveTriangleID> geometry_instance_to_triangle_id;
    /// Buffer containing mapping from EmissiveTriangleID to ActiveEmissiveTriangleID.
    /// Contains ActiveEmissiveTriangle.invalid if triangle is non-emissive.
    public StructuredBuffer<ActiveEmissiveTriangleID> triangle_id_to_active_triangle_id;
    /// Buffer containing mapping from ActiveEmissiveTriangleID to EmissiveTriangleID.
    public StructuredBuffer<EmissiveTriangleID> active_triangle_id_to_triangle_id;

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

    /// Get the area-averaged RGB radiant exitance of a triangle.
    public float3 get_triangle_emission(EmissiveTriangleID id) { return triangle_emission[uint(id)].rgb; }

    /// Get the luminance-weighted one-sided emitted radiant flux of a triangle.
    public float get_triangle_flux(EmissiveTriangleID id) { return triangle_flux[uint(id)]; }
}

/// Samples an emissive triangle uniformly by area and evaluates the probability density function.
/// Note that the triangle is only emitting on the front-facing side.
/// @param shading_pos_ws Shading point in world-space.
/// @param triangle_id Emissive triangle id.
/// @param u Uniform random number (2D).
/// @param[out] sample Sampled light, including emitted radiance divided by its PDF.
/// @return True if a sample was generated, false otherwise.
// [Differentiable]
public bool sample_emissive_triangle(
    no_diff const float3 shading_pos_ws,
    no_diff const EmissiveTriangleID triangle_id,
    no_diff const float2 u,
    out LightSample sample
)
{
    sample = {};

    sample.emitter_type = EmitterType::emissive_triangle;
    sample.emitter_id = uint(triangle_id);

    const EmissiveTriangle triangle = g_scene.emissive_geometry.get_triangle(triangle_id);

    // Sample the triangle uniformly.
    const float3 barycentrics = sample_triangle(u);
    sample.pos_ws = triangle.get_pos_ws(barycentrics);

    // Compute light vector and squared distance.
    float3 to_light = sample.pos_ws - shading_pos_ws;
    const float dist_sqr = max(float.MIN, dot(to_light, to_light));
    sample.distance = sqrt(dist_sqr);
    sample.dir_ws = to_light / sample.distance;

    // Get triangle normal.
    sample.normal_ws = triangle.normal_ws;

    // Reject back-facing triangle.
    float cos_theta = dot(sample.normal_ws, -sample.dir_ws);
    if (cos_theta <= 0.f)
        return false;

    // Compute probability density with respect to solid angle from the shading point.
    float denom = max(float.MIN, (cos_theta * triangle.area_ws));
    sample.pdf = dist_sqr / denom;

    // Evaluate emitted radiance.
    let material = g_scene.get_material(triangle.material_id);
    SurfaceInteraction si = {};
    si.wi_ws = -sample.dir_ws;
    si.uv = triangle.get_uv(barycentrics);
    sample.weight = material.eval_emissive_radiance(si, ExplicitLodSampler()) / sample.pdf;

    return true;
}

/// Evaluates the PDF for a light sample given a hit point on an emissive triangle.
/// @param ctx The shading point context.
/// @param triangle_id Emissive triangle ID.
/// @param emitter_pos_ws Sampled emitter position in world-space.
/// @param emitter_normal_ws Geometric normal at the sampled emitter position.
/// @return Probability density with respect to solid angle at the shading point.
public float eval_emissive_triangle_pdf(
    const LightSampleContext ctx,
    const EmissiveTriangleID triangle_id,
    const float3 emitter_pos_ws,
    const float3 emitter_normal_ws
)
{
    // Compute light vector and squared distance.
    float3 to_light = emitter_pos_ws - ctx.pos_ws;
    const float dist_sqr = max(float.MIN, dot(to_light, to_light));
    float3 dir = to_light / sqrt(dist_sqr);

    // Reject back-facing triangle.
    float cos_theta = dot(emitter_normal_ws, -dir);
    if (cos_theta <= 0.f)
        return 0.f;

    // Compute probability density with respect to solid angle from the shading point.
    EmissiveTriangle triangle = g_scene.emissive_geometry.get_triangle(triangle_id);
    float denom = max(float.MIN, (cos_theta * triangle.area_ws));
    return dist_sqr / denom;
}
