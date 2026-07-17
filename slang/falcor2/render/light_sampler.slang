// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Sampler for scene light sources.
/// Combines multiple emitter types (analytic lights, environment light, emissive triangles)
/// into a single sampling strategy. First selects an emitter type with uniform probability
/// among enabled and present types, then delegates to the type-specific sampler.
/// @tparam ENABLE_ANALYTIC_LIGHTS Enable sampling of analytic (point, directional, etc.) lights.
/// @tparam ENABLE_ENVIRONMENT_LIGHT Enable sampling of the environment (infinite area) light.
/// @tparam ENABLE_EMISSIVE_TRIANGLES Enable sampling of emissive triangle geometry.
public struct LightSampler<
    let ENABLE_ANALYTIC_LIGHTS = true,
    let ENABLE_ENVIRONMENT_LIGHT = true,
    let ENABLE_EMISSIVE_TRIANGLES = false>
{
    /// Number of supported emitter types (analytic, environment, emissive triangles).
    static const uint EMITTER_TYPE_COUNT = 3;

    /// Per-type selection probabilities, one entry per emitter type.
    typealias TypeProbabilities = vector<float, EMITTER_TYPE_COUNT>;

    /// Sampler used when emissive triangles are enabled. Selects triangles proportional to power.
    public PowerEmissiveTriangleSampler emissive_triangle_sampler;

    /// Samples a direction toward a light source from the given interaction point.
    /// First randomly selects an emitter type, then samples a direction from that type.
    /// @param ref The interaction (shading point) from which to sample.
    /// @param sg Sample generator.
    /// @param[out] sample The resulting direction sample.
    /// @param[out] weight The sample weight (incident radiance divided by the full PDF).
    /// @return True if a valid sample was generated, false otherwise.
    public bool sample_direction<Interaction : IInteraction, SampleGenerator : ISampleGenerator>(
        const Interaction ref,
        inout SampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    )
    {
        sample = {};
        weight = {};

        float type_pdf = 0.0;
        if (!sample_type(sg, sample.emitter_type, type_pdf))
            return false;

        if (ENABLE_ANALYTIC_LIGHTS && sample.emitter_type == EmitterType::analytic) {
            uint index = 0;
            float selection_pdf = 0.0;
            if (!g_scene.light_system.analytic_light_selection_distribution
                     .sample(next_float(sg), index, selection_pdf))
                return false;
            let light = g_scene.light_system.get_light(LightID(index));
            if (!light.sample_direction(ref, sg, sample, weight))
                return false;
            sample.pdf *= selection_pdf;
            weight /= selection_pdf;
        } else if (ENABLE_ENVIRONMENT_LIGHT && sample.emitter_type == EmitterType::environment) {
            if (!sample_environment_direction(ref, sg, sample, weight))
                return false;
        } else if (ENABLE_EMISSIVE_TRIANGLES && sample.emitter_type == EmitterType::emissive_triangle) {
            if (!emissive_triangle_sampler.sample_direction(ref, sg, sample, weight))
                return false;
        } else {
            return false;
        }

        sample.pdf *= type_pdf;
        weight /= type_pdf;
        return true;
    }

    /// Evaluates the probability density function for a given direction sample.
    /// Combines the emitter type selection probability with the type-specific PDF.
    /// Analytic lights always return 0 (they are delta distributions).
    /// @param ref The interaction (shading point) from which the sample was taken.
    /// @param sample The direction sample to evaluate.
    /// @return The PDF value with respect to solid angle at the shading point.
    public float eval_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        float type_pdf = eval_type_pdf(sample.emitter_type);
        if (ENABLE_ANALYTIC_LIGHTS && sample.emitter_type == EmitterType::analytic) {
            return 0.0;
        } else if (ENABLE_ENVIRONMENT_LIGHT && sample.emitter_type == EmitterType::environment) {
            return type_pdf * eval_environment_direction_pdf(ref, sample);
        } else if (ENABLE_EMISSIVE_TRIANGLES && sample.emitter_type == EmitterType::emissive_triangle) {
            return type_pdf * emissive_triangle_sampler.eval_direction_pdf(ref, sample);
        }
        return 0.0;
    }

    /// Sample one environment light from their power-weighted mixture.
    /// The single-light path deliberately consumes no additional random number.
    public bool sample_environment_direction<Interaction : IInteraction, SampleGenerator : ISampleGenerator>(
        const Interaction ref,
        inout SampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    )
    {
        const uint count = g_scene.light_system.environment_light_count;
        if (count == 0)
            return false;
        if (count == 1)
            return g_scene.light_system.get_environment_light(0).sample_direction(ref, sg, sample, weight);

        uint index = 0;
        float selection_pdf = 0.0;
        if (!g_scene.light_system.environment_light_selection_distribution.sample(next_float(sg), index, selection_pdf))
            return false;

        let light = g_scene.light_system.get_environment_light(index);
        if (!light.sample_direction(ref, sg, sample, weight))
            return false;

        sample.pdf = eval_environment_direction_pdf(ref, sample);
        weight = g_scene.light_system.eval_environment_direction(ref, sample) / sample.pdf;
        return true;
    }

    /// Evaluate the PDF of the power-weighted mixture of all environment lights.
    public float
    eval_environment_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        const uint count = g_scene.light_system.environment_light_count;
        if (count == 0)
            return 0.0;
        if (count == 1)
            return g_scene.light_system.get_environment_light(0).eval_direction_pdf(ref, sample);

        float pdf = 0.0;
        for (uint index = 0; index < count; ++index) {
            const float selection_pdf = g_scene.light_system.environment_light_selection_distribution.eval_pdf(index);
            pdf += selection_pdf * g_scene.light_system.get_environment_light(index).eval_direction_pdf(ref, sample);
        }
        return pdf;
    }

    /// Computes the selection probabilities for each emitter type.
    /// Enabled types that are present in the scene receive equal probability;
    /// disabled or absent types receive zero.
    /// @return A vector of normalized probabilities, one per emitter type.
    public TypeProbabilities compute_type_probabilities()
    {
        const float analytic_lights = (ENABLE_ANALYTIC_LIGHTS && g_scene.has_analytic_lights()) ? 1.0 : 0.0;
        const float environment_light = (ENABLE_ENVIRONMENT_LIGHT && g_scene.has_environment_lights()) ? 1.0 : 0.0;
        const float emissive_triangles = (ENABLE_EMISSIVE_TRIANGLES && g_scene.has_emissive_triangles()) ? 1.0 : 0.0;
        const float total = analytic_lights + environment_light + emissive_triangles;
        const float inv_total = total > 0.0 ? 1.0 / total : 0.0;
        return TypeProbabilities(analytic_lights, environment_light, emissive_triangles) * inv_total;
    }

    /// Randomly selects an emitter type based on the computed type probabilities.
    /// @param sg Random sample generator, advanced during sampling.
    /// @param[out] type The selected emitter type.
    /// @param[out] pdf The probability of selecting the returned type.
    /// @return True if a valid type was selected, false if no emitter types are available.
    private bool
    sample_type<SampleGenerator : ISampleGenerator>(inout SampleGenerator sg, out EmitterType type, out float pdf)
    {
        TypeProbabilities p = compute_type_probabilities();
        float x = next_float(sg);
        if (ENABLE_ANALYTIC_LIGHTS && x < p[0]) {
            type = EmitterType::analytic;
            pdf = p[0];
            return true;
        } else if (ENABLE_ENVIRONMENT_LIGHT && x < p[0] + p[1]) {
            type = EmitterType::environment;
            pdf = p[1];
            return true;
        } else if (ENABLE_EMISSIVE_TRIANGLES) {
            type = EmitterType::emissive_triangle;
            pdf = p[2];
            return true;
        } else {
            type = EmitterType::unknown;
            pdf = 0.0;
            return false;
        }
    }

    /// Evaluates the selection probability for a given emitter type.
    /// @param type The emitter type to query.
    /// @return The probability of selecting the given type, or 0 if it is disabled/absent.
    private float eval_type_pdf(EmitterType type)
    {
        TypeProbabilities p = compute_type_probabilities();
        if (ENABLE_ANALYTIC_LIGHTS && type == EmitterType::analytic) {
            return p[0];
        } else if (ENABLE_ENVIRONMENT_LIGHT && type == EmitterType::environment) {
            return p[1];
        } else if (ENABLE_EMISSIVE_TRIANGLES && type == EmitterType::emissive_triangle) {
            return p[2];
        } else {
            return 0.0;
        }
    }
}
