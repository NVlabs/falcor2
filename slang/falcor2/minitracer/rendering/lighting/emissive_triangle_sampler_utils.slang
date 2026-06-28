// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing rendering;

/// Samples an emissive triangle uniformly by area and evaluates the probability density function.
/// Note that the triangle is only emitting on the front-facing side.
/// @param pos_ws Shading point in world-space.
/// @param triangle Emissive triangle to sample.
/// @param u Uniform random number (2D).
/// @param[out] sample Sampled direction.
/// @param[out] weight Sample weight (emitted radiance divided by pdf).
/// @return True if a sample was generated, false otherwise.
[Differentiable]
public bool sample_emissive_triangle(
    no_diff const float3 pos_ws,
    no_diff const EmissiveTriangleID triangle,
    no_diff const float2 u,
    no_diff out DirectionSample sample,
    out float3 weight
)
{
    sample = {};
    weight = {};

    sample.emitter_type = EmitterType::emissive_triangle;
    sample.primitive_id = uint(triangle);

    const EmissiveTriangle tri = g_scene.emissive_triangles.get_triangle(triangle);

    // Sample the triangle uniformly.
    const float3 barycentrics = sample_triangle(u);
    sample.pos_ws = tri.get_pos_ws(barycentrics);

    // Compute light vector and squared distance.
    float3 to_light = sample.pos_ws - pos_ws;
    // Clamp to avoid NaNs below.
    const float dist_sqr = max(float.MIN, dot(to_light, to_light));
    sample.distance = sqrt(dist_sqr);
    // Note: to_light can be zero.
    sample.dir_ws = to_light / sample.distance;

    // Get triangle normal.
    sample.normal_ws = tri.normal_ws;

    // Reject sample if triangle is back-facing.
    float cos_theta = dot(sample.normal_ws, -sample.dir_ws);
    if (cos_theta <= 0.f)
        return false;

    // Compute probability density with respect to solid angle from the shading point.
    // The farther away the light is and the larger the angle it is at, the larger the pdf becomes.
    // The probability goes to infinity in the limit. Note: Guard against div-by-zero here by
    // clamping.
    float denom = max(float.MIN, cos_theta* tri.area_ws);
    sample.pdf = dist_sqr / denom;

    // Evaluate emitted radiance.
    float2 uv = tri.get_uv(barycentrics);
    weight = g_scene.eval_emitted_radiance(tri.material, uv) / sample.pdf;

    // Save light sample triangle barycentric coordinates
    sample.uv = u;

    // TODO: We can simplify the expressions by using the unnormalized quantities for computing the
    // pdf.
    // sample.pdf = -2.f * dist_sqr * sample.distance / (dot(N, L);    // Optimized (except N would
    // have to be properly flipped above, before normal_ws is computed).
    return true;
}

/// Evaluates the PDF for a light sample given a hit point on an emissive triangle.
/// @param pos_ws Shading point in world space.
/// @param sample Sampled direction.
/// @return Probability density with respect to solid angle at the shading point.
public float eval_emissive_triangle_pdf(const float3 pos_ws, const DirectionSample sample)
{
    // Compute light vector and squared distance.
    float3 to_light = sample.pos_ws - pos_ws; // Unnormalized light vector
    const float dist_sqr = dot(to_light, to_light);
    if (dist_sqr <= float.MIN)
        return 0.f; // Avoid NaNs below
    float3 L = to_light / sqrt(dist_sqr);

    // Cosine of angle between the light's normal and the light vector (flip L since it points
    // towards the light).
    float cos_theta = dot(sample.normal_ws, -L);
    if (cos_theta <= 0.f)
        return 0.f;

    // Compute probability density with respect to solid angle from the shading point.
    // The farther away the light is and the larger the angle it is at, the larger the pdf becomes.
    // The probability goes to infinity in the limit. Note: Guard against div-by-zero here by
    // clamping.
    // TODO: Do we need the clamp here? dist_sqr is already clamped, so NaN should not be possible
    // (but +inf is).
    float denom = max(
        float.MIN,
        cos_theta* g_scene.emissive_triangles.get_triangle(EmissiveTriangleID(sample.primitive_id)).area_ws
    );
    return dist_sqr / denom;
}
