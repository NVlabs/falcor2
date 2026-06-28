// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing rendering;

/// Sampler for emissive triangles in the scene.
/// Uniformly picks an emissive triangle and samples it uniformly by area.
public struct UniformEmissiveTriangleSampler
{
    [Differentiable]
    public bool sample(
        no_diff const Interaction interaction,
        no_diff inout UniformSampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    )
    {
        sample = {};
        weight = {};

        uint triangle_count = g_scene.emissive_triangles.triangle_count;

        if (triangle_count == 0)
            return false;

        // Pick a random triangle uniformly.
        float u_selection = next_float(sg);
        uint triangle_index = min(uint(u_selection * triangle_count), triangle_count - 1);
        float selection_pdf = 1.0 / float(triangle_count);

        // Sample the triangle uniformly.
        float2 u = next_float2(sg);
        if (!sample_emissive_triangle(interaction.pos_ws, EmissiveTriangleID(triangle_index), u, sample, weight))
            return false;

        // The final probability density is the product of the sampling probabilities.
        sample.pdf *= selection_pdf;
        weight *= triangle_count; // same as /= selection_pdf;

        return true;
    }

    public float eval_pdf(const Interaction interaction, const DirectionSample sample)
    {
        EmissiveTriangleID triangle_id = EmissiveTriangleID(sample.primitive_id);
        uint triangle_count = g_scene.emissive_triangles.triangle_count;

        if (sample.emitter_type != EmitterType::emissive_triangle || triangle_id == EmissiveTriangleID::invalid
            || triangle_count == 0)
            return 0.0;

        float selection_pdf = 1.0 / float(triangle_count);

        // Compute triangle sampling probability with respect to solid angle from the shading point.
        float triangle_pdf = eval_emissive_triangle_pdf(interaction.pos_ws, sample);

        // The final probability density is the product of the sampling probabilities.
        return selection_pdf * triangle_pdf;
    }
}
