// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Interface for sampling emissive triangles in the scene.
public interface IEmissiveTriangleSampler
{
    /// Samples a direction toward an emissive triangle from the given interaction point.
    /// @param ref The interaction (shading point) from which to sample.
    /// @param sg Sample generator.
    /// @param[out] sample The resulting direction sample describing the sampled point on the emissive triangle.
    /// @param[out] weight The sample weight (emitted radiance divided by the full PDF).
    /// @return True if a valid sample was generated, false otherwise.
    public bool sample_direction<Interaction : IInteraction, SampleGenerator : ISampleGenerator>(
        const Interaction ref,
        inout SampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    );

    /// Evaluates the probability density function for a given direction sample.
    /// @param ref The interaction (shading point) from which the sample was taken.
    /// @param sample The direction sample to evaluate.
    /// @return The PDF value with respect to solid angle at the shading point.
    public float eval_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample);
}

/// Emissive triangle sampler that selects triangles with uniform probability.
/// Each active emissive triangle has an equal chance of being selected, regardless of
/// its emitted power or surface area.
public struct UniformEmissiveTriangleSampler : IEmissiveTriangleSampler
{
    public bool sample_direction<Interaction : IInteraction, SampleGenerator : ISampleGenerator>(
        const Interaction ref,
        inout SampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    )
    {
        sample = {};
        weight = {};

        uint active_triangle_count = g_scene.emissive_geometry.active_triangle_count;

        // Reject if there are no active emissive triangles.
        if (active_triangle_count == 0)
            return false;

        // Pick a random triangle uniformly.
        uint active_triangle_index = min(uint(next_float(sg) * active_triangle_count), active_triangle_count - 1);
        float selection_pdf = 1.0 / float(active_triangle_count);

        // Get the EmissiveTriangleID of the sampled triangle.
        EmissiveTriangleID triangle_id
            = g_scene.emissive_geometry.active_triangle_id_to_triangle_id[active_triangle_index];

        // Sample the triangle uniformly.
        float2 u = next_float2(sg);
        if (!sample_emissive_triangle(ref.position_ws, triangle_id, u, sample, weight))
            return false;

        // The final probability density is the product of the sampling probabilities.
        sample.pdf *= selection_pdf;
        weight *= active_triangle_count; // same as /= selection_pdf;

        return true;
    }

    public float eval_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        EmissiveTriangleID triangle_id = EmissiveTriangleID(sample.primitive_id);
        uint active_triangle_count = g_scene.emissive_geometry.active_triangle_count;

        // Reject if there are no active emissive triangles, or if the sample is not from an emissive triangle.
        if (active_triangle_count == 0 || sample.emitter_type != EmitterType::emissive_triangle
            || triangle_id == EmissiveTriangleID::invalid)
            return 0.0;

        // Reject if the triangle is not active.
        if (g_scene.emissive_geometry.triangle_id_to_active_triangle_id[uint(triangle_id)]
            == ActiveEmissiveTriangleID::invalid)
            return 0.0;

        // Triangle selection probability.
        float selection_pdf = 1.0 / float(active_triangle_count);

        // Compute triangle sampling probability with respect to solid angle from the shading point.
        float triangle_pdf = eval_emissive_triangle_pdf(ref.position_ws, sample);

        // The final probability density is the product of the sampling probabilities.
        return selection_pdf * triangle_pdf;
    }
}

/// Emissive triangle sampler that selects triangles proportional to their emitted power.
/// Uses a pre-built alias table to efficiently sample triangles weighted by their total
/// emitted flux (emission * area).
public struct PowerEmissiveTriangleSampler : IEmissiveTriangleSampler
{
    public bool sample_direction<Interaction : IInteraction, SampleGenerator : ISampleGenerator>(
        const Interaction ref,
        inout SampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    )
    {
        sample = {};
        weight = {};

        uint active_triangle_count = g_scene.emissive_geometry.active_triangle_count;

        // Reject if there are no active emissive triangles.
        if (active_triangle_count == 0)
            return false;

        // Pick a random triangle using the alias table.
        uint active_triangle_index = 0;
        float selection_pdf = 0.0;
        if (!g_scene.emissive_geometry.active_triangle_alias_table
                 .sample(next_float(sg), active_triangle_index, selection_pdf))
            return false;

        // Get the EmissiveTriangleID of the sampled triangle.
        EmissiveTriangleID triangle_id
            = g_scene.emissive_geometry.active_triangle_id_to_triangle_id[active_triangle_index];

        // Sample the triangle uniformly.
        float2 u = next_float2(sg);
        if (!sample_emissive_triangle(ref.position_ws, triangle_id, u, sample, weight))
            return false;

        // The final probability density is the product of the sampling probabilities.
        sample.pdf *= selection_pdf;
        weight /= selection_pdf;

        return true;
    }

    public float eval_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        EmissiveTriangleID triangle_id = EmissiveTriangleID(sample.primitive_id);
        uint active_triangle_count = g_scene.emissive_geometry.active_triangle_count;

        // Reject if there are no active emissive triangles, or if the sample is not from an emissive triangle.
        if (active_triangle_count == 0 || sample.emitter_type != EmitterType::emissive_triangle
            || triangle_id == EmissiveTriangleID::invalid)
            return 0.0;

        ActiveEmissiveTriangleID active_triangle_id
            = g_scene.emissive_geometry.triangle_id_to_active_triangle_id[uint(triangle_id)];

        // Reject if the triangle is not active.
        if (active_triangle_id == ActiveEmissiveTriangleID::invalid)
            return 0.0;

        // Triangle selection probability.
        uint active_triangle_index = uint(active_triangle_id);
        float selection_pdf = g_scene.emissive_geometry.active_triangle_alias_table.eval_pdf(active_triangle_index);

        // Compute triangle sampling probability with respect to solid angle from the shading point.
        float triangle_pdf = eval_emissive_triangle_pdf(ref.position_ws, sample);

        // The final probability density is the product of the sampling probabilities.
        return selection_pdf * triangle_pdf;
    }
}

/// Samples an emissive triangle uniformly by area and evaluates the probability density function.
/// Note that the triangle is only emitting on the front-facing side.
/// @param pos_ws Shading point in world-space.
/// @param triangle_id Emissive triangle id.
/// @param u Uniform random number (2D).
/// @param[out] sample Sampled direction.
/// @param[out] weight Sample weight (emitted radiance divided by pdf).
/// @return True if a sample was generated, false otherwise.
// [Differentiable]
public bool sample_emissive_triangle(
    no_diff const float3 pos_ws,
    no_diff const EmissiveTriangleID triangle_id,
    no_diff const float2 u,
    no_diff out DirectionSample sample,
    out float3 weight
)
{
    sample = {};
    weight = {};

    sample.emitter_type = EmitterType::emissive_triangle;
    sample.primitive_id = uint(triangle_id);

    const EmissiveTriangle triangle = g_scene.emissive_geometry.get_triangle(triangle_id);

    // Sample the triangle uniformly.
    const float3 barycentrics = sample_triangle(u);
    sample.pos_ws = triangle.get_pos_ws(barycentrics);

    // Compute light vector and squared distance.
    float3 to_light = sample.pos_ws - pos_ws;
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
    weight = material.eval_emissive_radiance(si, ExplicitLodSampler()) / sample.pdf;

    // Save light sample triangle barycentric coordinates
    sample.uv = u;

    return true;
}

/// Evaluates the PDF for a light sample given a hit point on an emissive triangle.
/// @param pos_ws Shading point in world space.
/// @param sample Sampled direction.
/// @return Probability density with respect to solid angle at the shading point.
public float eval_emissive_triangle_pdf(const float3 pos_ws, const DirectionSample sample)
{
    // Compute light vector and squared distance.
    float3 to_light = sample.pos_ws - pos_ws;
    const float dist_sqr = max(float.MIN, dot(to_light, to_light));
    float3 dir = to_light / sqrt(dist_sqr);

    // Reject back-facing triangle.
    float cos_theta = dot(sample.normal_ws, -dir);
    if (cos_theta <= 0.f)
        return 0.f;

    // Compute probability density with respect to solid angle from the shading point.
    EmissiveTriangle triangle = g_scene.emissive_geometry.get_triangle(EmissiveTriangleID(sample.primitive_id));
    float denom = max(float.MIN, (cos_theta * triangle.area_ws));
    return dist_sqr / denom;
}
