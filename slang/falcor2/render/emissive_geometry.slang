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

/// Construct the complete surface interaction for a point on an emissive triangle.
public SurfaceInteraction make_emissive_triangle_surface_interaction(
    const EmissiveTriangleID triangle_id,
    const float3 barycentrics,
    const float3 wi_ws
)
{
    const EmissiveTriangle triangle = g_scene.emissive_geometry.get_triangle(triangle_id);
    const EmissiveTriangleID first_triangle_id
        = g_scene.emissive_geometry.geometry_instance_to_triangle_id[uint(triangle.geometry_instance_id)];
    const uint primitive_index = uint(triangle_id) - uint(first_triangle_id);
    const TriangleHitInfo hit = TriangleHitInfo(triangle.geometry_instance_id, primitive_index, barycentrics.yz);
    return g_scene.make_surface_interaction(hit, wi_ws);
}

/// Samples an emissive triangle uniformly by area and evaluates the probability density function.
/// Note that the triangle is only emitting on the front-facing side.
/// @param shading_pos_ws Shading point in world-space.
/// @param triangle_id Emissive triangle id.
/// @param u Uniform sample in [0, 1)^2.
/// @param[out] sample Sampled light, including emitted radiance divided by its PDF.
/// @param[out] sample_ref Triangle identity and sampled barycentric coordinates; retain only on success.
/// @return True if a sample was generated, false otherwise.
// [Differentiable]
public bool sample_emissive_triangle(
    no_diff const float3 shading_pos_ws,
    no_diff const EmissiveTriangleID triangle_id,
    no_diff const float2 u,
    out LightSample sample,
    out LightSampleRef sample_ref
)
{
    sample = {};
    sample_ref = {};

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
    const SurfaceInteraction si = make_emissive_triangle_surface_interaction(triangle_id, barycentrics, -sample.dir_ws);
    sample.radiance = material.eval_emissive_radiance(si, ExplicitLodSampler());
    sample.weight = sample.radiance / sample.pdf;
    sample.flags = LightSampleFlags::has_surface_point;

    sample_ref.emitter_type = EmitterType::emissive_triangle;
    sample_ref.emitter_id = uint(triangle_id);
    sample_ref.parameters = barycentrics.yz;

    return true;
}

/// Compatibility overload that discards the retained triangle endpoint.
public bool sample_emissive_triangle(
    no_diff const float3 shading_pos_ws,
    no_diff const EmissiveTriangleID triangle_id,
    no_diff const float2 u,
    out LightSample sample
)
{
    LightSampleRef sample_ref;
    return sample_emissive_triangle(shading_pos_ws, triangle_id, u, sample, sample_ref);
}

/// Reconstructs a retained triangle endpoint and re-evaluates its material emission toward the receiver.
public bool
eval_emissive_triangle_sample(const LightSampleContext ctx, const LightSampleRef sample_ref, out LightSample sample)
{
    sample = {};
    const float2 p = sample_ref.parameters;
    const EmissiveTriangleID triangle_id = EmissiveTriangleID(sample_ref.emitter_id);
    const EmissiveTriangle triangle = g_scene.emissive_geometry.get_triangle(triangle_id);
    const float3 barycentrics = float3(1.0 - p.x - p.y, p);
    sample.pos_ws = triangle.get_pos_ws(barycentrics);
    sample.normal_ws = triangle.normal_ws;
    const float3 to_light = sample.pos_ws - ctx.pos_ws;
    const float distance_squared = dot(to_light, to_light);
    if (distance_squared <= 0.0)
        return false;
    sample.distance = sqrt(distance_squared);
    sample.dir_ws = to_light / sample.distance;
    sample.emitter_type = EmitterType::emissive_triangle;
    sample.emitter_id = sample_ref.emitter_id;
    sample.flags = LightSampleFlags::has_surface_point;
    sample.pdf = eval_emissive_triangle_pdf(ctx, triangle_id, sample.pos_ws, sample.normal_ws);
    if (dot(sample.normal_ws, -sample.dir_ws) > 0.0) {
        const SurfaceInteraction si
            = make_emissive_triangle_surface_interaction(triangle_id, barycentrics, -sample.dir_ws);
        sample.radiance = g_scene.get_material(triangle.material_id).eval_emissive_radiance(si, ExplicitLodSampler());
    }
    sample.weight = sample.pdf > 0.0 ? sample.radiance / sample.pdf : float3(0.0);
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
