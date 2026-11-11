// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;

/// Uniform sample coordinates for scene-light selection and sampling.
public struct LightSampleInput
{
    /// Uniform sample in [0, 1) for selecting an emitter, including any hierarchical choices.
    public float u_select;
    /// Uniform sample in [0, 1)^2 for sampling a position or direction on the selected emitter.
    public float2 u_light;
}

/// Consume exactly three scalar dimensions in field order, even for deterministic lights.
public LightSampleInput make_light_sample_input<S : ISampleGenerator>(inout S sg)
{
    LightSampleInput input;
    input.u_select = next_float(sg);
    input.u_light = next_float2(sg);
    return input;
}

/// Interface for complete scene-light sampling strategies.
public interface ILightSampler
{
    /// Samples incident illumination from any active scene light source.
    public bool sample(
        const LightSampleContext ctx,
        const LightSampleInput input,
        out LightSample sample,
        out LightSampleRef sample_ref
    );

    /// Compatibility overload that discards the reference. Proxy samples can succeed with an invalid reference.
    public bool sample(const LightSampleContext ctx, const LightSampleInput input, out LightSample sample)
    {
        LightSampleRef sample_ref;
        return this.sample(ctx, input, sample, sample_ref);
    }

    /// Reconstructs a retained endpoint and computes its complete proposal density at the supplied context.
    /// Requires a valid scene-level reference; emitter identity and parameters are trusted.
    /// Returns true for valid zero-contribution or zero-density samples. No visibility or reservoir weight is reused.
    /// Returns false for degenerate connections or unsupported receiver contexts.
    /// Scene identities/parameterizations must be unchanged since sampling; virtual sphere proxies are unsupported.
    public bool eval_sample(const LightSampleContext ctx, const LightSampleRef sample_ref, out LightSample sample)
    {
        sample = {};
        if (sample_ref.emitter_type == EmitterType::analytic) {
            const LightID light_id = LightID(sample_ref.emitter_id);
            if (!g_scene.light_system.get_light(light_id).eval_sample(ctx, sample_ref, sample))
                return false;
            if (sample.has_flag(LightSampleFlags::has_surface_point))
                sample.pdf = this.eval_area_light_pdf(ctx, light_id, sample.pos_ws, sample.normal_ws);
            else
                sample.pdf *= this.eval_analytic_selection_pdf(ctx, light_id);
        } else if (sample_ref.emitter_type == EmitterType::environment) {
            const float3 dir = oct_to_ndir_unorm(sample_ref.parameters);
            sample.dir_ws = dir;
            sample.distance = float.MAX;
            sample.emitter_type = EmitterType::environment;
            sample.emitter_id = sample_ref.emitter_id;
            sample.radiance = g_scene.light_system.eval_environment_radiance(ctx, dir);
            sample.pdf = this.eval_environment_pdf(ctx, dir);
        } else if (sample_ref.emitter_type == EmitterType::emissive_triangle) {
            if (!eval_emissive_triangle_sample(ctx, sample_ref, sample))
                return false;
            sample.pdf = this.eval_emissive_pdf(
                ctx,
                EmissiveTriangleID(sample_ref.emitter_id),
                sample.pos_ws,
                sample.normal_ws
            );
        } else {
            // Unknown emitter type
            return false;
        }
        sample.weight = sample.pdf > 0.0 ? sample.radiance / sample.pdf : float3(0.0);
        return true;
    }

    /// Discrete probability of selecting a known analytic light, including emitter-domain selection.
    public float eval_analytic_selection_pdf(const LightSampleContext ctx, const LightID light_id);

    /// Evaluates the complete scene-light PDF for an environment direction.
    public float eval_environment_pdf(const LightSampleContext ctx, const float3 dir_ws);

    /// Evaluates the complete scene-light PDF for a known area-light hit.
    public float eval_area_light_pdf(
        const LightSampleContext ctx,
        const LightID light_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    );

    /// Evaluates the complete scene-light PDF for a known emissive triangle hit.
    public float eval_emissive_pdf(
        const LightSampleContext ctx,
        const EmissiveTriangleID triangle_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    );
}

namespace detail {

/// Probability of selecting the finite-emitter domain when both domains have positive estimates.
static const float FINITE_EMITTER_SELECTION_PROBABILITY = 0.5;

/// Marginal probabilities of selecting each emitter type before conditional sampling.
struct EmitterTypePMF
{
    float analytic;
    float environment;
    float emissive_triangle;
};

EmitterTypePMF compute_emitter_type_pmf(
    const float analytic_light_power,
    const float environment_light_power,
    const float emissive_triangle_power
)
{
    EmitterTypePMF pmf = {};
    const bool has_finite_emitters = analytic_light_power > 0.0 || emissive_triangle_power > 0.0;
    const bool has_environment_lights = environment_light_power > 0.0;

    if (!has_finite_emitters) {
        pmf.environment = has_environment_lights ? 1.0 : 0.0;
        return pmf;
    }

    const float finite_probability = has_environment_lights ? FINITE_EMITTER_SELECTION_PROBABILITY : 1.0;
    const float finite_power_scale = max(analytic_light_power, emissive_triangle_power);
    const float analytic_power = analytic_light_power / finite_power_scale;
    const float emissive_power = emissive_triangle_power / finite_power_scale;
    const float normalized_finite_power = analytic_power + emissive_power;
    if (analytic_power <= emissive_power) {
        pmf.analytic = finite_probability * (analytic_power / normalized_finite_power);
        pmf.emissive_triangle = finite_probability - pmf.analytic;
    } else {
        pmf.emissive_triangle = finite_probability * (emissive_power / normalized_finite_power);
        pmf.analytic = finite_probability - pmf.emissive_triangle;
    }
    pmf.environment = has_environment_lights ? 1.0 - finite_probability : 0.0;
    return pmf;
}

} // namespace detail

/// Samples every scene light source with uniform probability.
public struct UniformLightSampler : ILightSampler
{
    uint get_source_count()
    {
        return g_scene.light_system.light_count + g_scene.emissive_geometry.active_triangle_count;
    }

    public bool sample(
        const LightSampleContext ctx,
        const LightSampleInput input,
        out LightSample sample,
        out LightSampleRef sample_ref
    )
    {
        sample = {};
        sample_ref = {};

        const uint source_count = get_source_count();
        if (source_count == 0)
            return false;

        const uint analytic_light_count = g_scene.light_system.analytic_light_count;
        const uint environment_light_count = g_scene.light_system.environment_light_count;

        const uint source_index = min(uint(input.u_select * source_count), source_count - 1);
        const float selection_pdf = 1.0 / float(source_count);

        if (source_index < analytic_light_count) {
            // Sample an analytic light.
            const LightID light_id = LightID(source_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, input.u_light, sample, sample_ref))
                return false;
            sample.emitter_id = uint(light_id);
        } else if (source_index < analytic_light_count + environment_light_count) {
            // Sample an environment light.
            const uint environment_light_index = source_index - analytic_light_count;
            const LightID light_id = LightID(uint(g_scene.light_system.environment_light_id) + environment_light_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, input.u_light, sample, sample_ref))
                return false;
            sample.emitter_id = uint(light_id);
            if (environment_light_count > 1) {
                sample.pdf = eval_environment_mixture_pdf(ctx, sample.dir_ws);
                if (sample.pdf <= 0.0)
                    return false;
                sample.radiance = g_scene.light_system.eval_environment_radiance(ctx, sample.dir_ws);
                sample.weight = sample.radiance / sample.pdf;
            }
        } else {
            // Sample an emissive triangle.
            const uint active_triangle_index = source_index - analytic_light_count - environment_light_count;
            const EmissiveTriangleID triangle_id
                = g_scene.emissive_geometry.active_triangle_id_to_triangle_id[active_triangle_index];
            if (!sample_emissive_triangle(ctx.pos_ws, triangle_id, input.u_light, sample, sample_ref))
                return false;
        }

        // Adjust the sample PDF and weight to account for the uniform selection probability.
        sample.pdf *= selection_pdf;
        sample.weight /= selection_pdf;

        // Propagate the emitter ID to the sample reference.
        sample_ref.emitter_id = sample.emitter_id;

        return true;
    }

    public float eval_analytic_selection_pdf(const LightSampleContext ctx, const LightID light_id)
    {
        return uint(light_id) < g_scene.light_system.analytic_light_count ? eval_source_selection_pdf() : 0.0;
    }

    /// Evaluates the selection PDF for a source known to be valid.
    public float eval_source_selection_pdf() { return 1.0 / float(get_source_count()); }

    float eval_environment_mixture_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        float pdf = 0.0;
        for (uint i = 0; i < g_scene.light_system.environment_light_count; ++i)
            pdf += g_scene.light_system.get_environment_light(i).eval_pdf(ctx, dir_ws);
        return pdf;
    }

    public float eval_environment_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        if (g_scene.light_system.environment_light_count == 0)
            return 0.0;
        return eval_source_selection_pdf() * eval_environment_mixture_pdf(ctx, dir_ws);
    }

    public float eval_area_light_pdf(
        const LightSampleContext ctx,
        const LightID light_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        if (uint(light_id) >= g_scene.light_system.analytic_light_count)
            return 0.0;
        return eval_source_selection_pdf()
            * g_scene.light_system.get_light(light_id)
                  .eval_surface_sampling_pdf(ctx, emitter_pos_ws, emitter_normal_ws);
    }

    public float eval_emissive_pdf(
        const LightSampleContext ctx,
        const EmissiveTriangleID triangle_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        if (uint(triangle_id) >= g_scene.emissive_geometry.triangle_count)
            return 0.0;
        const ActiveEmissiveTriangleID active_triangle_id
            = g_scene.emissive_geometry.triangle_id_to_active_triangle_id[uint(triangle_id)];
        if (active_triangle_id == ActiveEmissiveTriangleID::invalid)
            return 0.0;
        return eval_source_selection_pdf()
            * eval_emissive_triangle_pdf(ctx, triangle_id, emitter_pos_ws, emitter_normal_ws);
    }
}

/// Samples finite and infinite emitters with configured probability, then power-weights within each domain.
public struct PowerLightSampler : ILightSampler
{
    public AliasTable1D analytic_light_distribution;
    public AliasTable1D environment_light_distribution;
    public AliasTable1D emissive_triangle_distribution;
    /// Luminance-weighted emitted radiant flux of all analytic lights.
    public float analytic_light_power;
    /// Luminance-weighted integral of incident radiance over solid angle for all infinite lights.
    public float environment_light_power;
    /// Luminance-weighted emitted radiant flux of all emissive triangles.
    public float emissive_triangle_power;

    public bool sample(
        const LightSampleContext ctx,
        const LightSampleInput input,
        out LightSample sample,
        out LightSampleRef sample_ref
    )
    {
        sample = {};
        sample_ref = {};

        const detail::EmitterTypePMF emitter_type_pmf
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power);
        if (emitter_type_pmf.analytic <= 0.0 && emitter_type_pmf.environment <= 0.0
            && emitter_type_pmf.emissive_triangle <= 0.0)
            return false;

        float light_sample = input.u_select;
        float selection_pdf = 0.0;
        if (light_sample < emitter_type_pmf.analytic) {
            // Sample an analytic light.
            light_sample /= emitter_type_pmf.analytic;
            uint light_index;
            if (!analytic_light_distribution.sample(light_sample, light_index, selection_pdf))
                return false;
            selection_pdf *= emitter_type_pmf.analytic;
            if (selection_pdf <= 0.0)
                return false;

            const LightID light_id = LightID(light_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, input.u_light, sample, sample_ref))
                return false;
            sample.emitter_id = uint(light_id);
        } else if (light_sample < 1.0 - emitter_type_pmf.environment) {
            // Sample an emissive triangle.
            light_sample = (light_sample - emitter_type_pmf.analytic) / emitter_type_pmf.emissive_triangle;
            uint active_triangle_index;
            if (!emissive_triangle_distribution.sample(light_sample, active_triangle_index, selection_pdf))
                return false;
            selection_pdf *= emitter_type_pmf.emissive_triangle;
            if (selection_pdf <= 0.0)
                return false;

            const EmissiveTriangleID triangle_id
                = g_scene.emissive_geometry.active_triangle_id_to_triangle_id[active_triangle_index];
            if (!sample_emissive_triangle(ctx.pos_ws, triangle_id, input.u_light, sample, sample_ref))
                return false;
        } else {
            // Sample an environment light.
            light_sample = (light_sample - (1.0 - emitter_type_pmf.environment)) / emitter_type_pmf.environment;
            uint environment_light_index;
            if (!environment_light_distribution.sample(light_sample, environment_light_index, selection_pdf))
                return false;
            selection_pdf *= emitter_type_pmf.environment;
            if (selection_pdf <= 0.0)
                return false;

            const LightID light_id = LightID(uint(g_scene.light_system.environment_light_id) + environment_light_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, input.u_light, sample, sample_ref))
                return false;
            sample.emitter_id = uint(light_id);
            if (g_scene.light_system.environment_light_count > 1) {
                sample.pdf = eval_environment_mixture_pdf(ctx, sample.dir_ws);
                if (sample.pdf <= 0.0)
                    return false;
                sample.radiance = g_scene.light_system.eval_environment_radiance(ctx, sample.dir_ws);
                sample.weight = sample.radiance / sample.pdf;
                selection_pdf = emitter_type_pmf.environment;
            }
        }

        // Adjust the conditional sample PDF and weight for the applicable selection probability.
        sample.pdf *= selection_pdf;
        sample.weight /= selection_pdf;

        // Propagate the emitter ID to the sample reference.
        sample_ref.emitter_id = sample.emitter_id;

        return true;
    }

    public float eval_source_selection_pdf(const EmitterType type, const uint index)
    {
        const detail::EmitterTypePMF emitter_type_pmf
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power);
        if (type == EmitterType::analytic && emitter_type_pmf.analytic > 0.0)
            return emitter_type_pmf.analytic * analytic_light_distribution.eval_pdf(index);
        if (type == EmitterType::environment && emitter_type_pmf.environment > 0.0)
            return emitter_type_pmf.environment * environment_light_distribution.eval_pdf(index);
        if (type == EmitterType::emissive_triangle && emitter_type_pmf.emissive_triangle > 0.0)
            return emitter_type_pmf.emissive_triangle * emissive_triangle_distribution.eval_pdf(index);
        return 0.0;
    }

    public float eval_analytic_selection_pdf(const LightSampleContext ctx, const LightID light_id)
    {
        if (uint(light_id) >= g_scene.light_system.analytic_light_count)
            return 0.0;
        return eval_source_selection_pdf(EmitterType::analytic, uint(light_id));
    }

    float eval_environment_mixture_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        float pdf = 0.0;
        for (uint i = 0; i < g_scene.light_system.environment_light_count; ++i)
            pdf += environment_light_distribution.eval_pdf(i)
                * g_scene.light_system.get_environment_light(i).eval_pdf(ctx, dir_ws);
        return pdf;
    }

    public float eval_environment_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        const float environment_probability
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power)
                  .environment;
        if (environment_probability <= 0.0)
            return 0.0;
        return environment_probability * eval_environment_mixture_pdf(ctx, dir_ws);
    }

    public float eval_area_light_pdf(
        const LightSampleContext ctx,
        const LightID light_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        if (uint(light_id) >= g_scene.light_system.analytic_light_count)
            return 0.0;
        return eval_source_selection_pdf(EmitterType::analytic, uint(light_id))
            * g_scene.light_system.get_light(light_id)
                  .eval_surface_sampling_pdf(ctx, emitter_pos_ws, emitter_normal_ws);
    }

    public float eval_emissive_pdf(
        const LightSampleContext ctx,
        const EmissiveTriangleID triangle_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        if (uint(triangle_id) >= g_scene.emissive_geometry.triangle_count)
            return 0.0;
        const ActiveEmissiveTriangleID active_triangle_id
            = g_scene.emissive_geometry.triangle_id_to_active_triangle_id[uint(triangle_id)];
        if (active_triangle_id == ActiveEmissiveTriangleID::invalid)
            return 0.0;
        return eval_source_selection_pdf(EmitterType::emissive_triangle, uint(active_triangle_id))
            * eval_emissive_triangle_pdf(ctx, triangle_id, emitter_pos_ws, emitter_normal_ws);
    }
}

/// Uses power sampling for component lights and an EmissiveTriangleTree for emissive triangles.
public struct HierarchicalLightSampler : ILightSampler
{
    public AliasTable1D analytic_light_distribution;
    public AliasTable1D environment_light_distribution;
    public float analytic_light_power;
    public float environment_light_power;
    public float emissive_triangle_power;
    public EmissiveTriangleTree emissive_triangle_tree;

    public bool sample(
        const LightSampleContext ctx,
        const LightSampleInput input,
        out LightSample sample,
        out LightSampleRef sample_ref
    )
    {
        sample = {};
        sample_ref = {};

        const detail::EmitterTypePMF emitter_type_pmf
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power);
        if (emitter_type_pmf.analytic <= 0.0 && emitter_type_pmf.environment <= 0.0
            && emitter_type_pmf.emissive_triangle <= 0.0)
            return false;

        float light_sample = input.u_select;
        float selection_pdf = 0.0;
        if (light_sample < emitter_type_pmf.analytic) {
            light_sample /= emitter_type_pmf.analytic;
            uint light_index;
            if (!analytic_light_distribution.sample(light_sample, light_index, selection_pdf))
                return false;
            selection_pdf *= emitter_type_pmf.analytic;
            if (selection_pdf <= 0.0)
                return false;

            const LightID light_id = LightID(light_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, input.u_light, sample, sample_ref))
                return false;
            sample.emitter_id = uint(light_id);
        } else if (light_sample < 1.0 - emitter_type_pmf.environment) {
            light_sample = (light_sample - emitter_type_pmf.analytic) / emitter_type_pmf.emissive_triangle;
            EmissiveTriangleID triangle_id;
            if (!emissive_triangle_tree.sample(ctx, light_sample, selection_pdf, triangle_id))
                return false;
            selection_pdf *= emitter_type_pmf.emissive_triangle;
            if (selection_pdf <= 0.0
                || !sample_emissive_triangle(ctx.pos_ws, triangle_id, input.u_light, sample, sample_ref))
                return false;
        } else {
            light_sample = (light_sample - (1.0 - emitter_type_pmf.environment)) / emitter_type_pmf.environment;
            uint environment_light_index;
            if (!environment_light_distribution.sample(light_sample, environment_light_index, selection_pdf))
                return false;
            selection_pdf *= emitter_type_pmf.environment;
            if (selection_pdf <= 0.0)
                return false;

            const LightID light_id = LightID(uint(g_scene.light_system.environment_light_id) + environment_light_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, input.u_light, sample, sample_ref))
                return false;
            sample.emitter_id = uint(light_id);
            if (g_scene.light_system.environment_light_count > 1) {
                sample.pdf = eval_environment_mixture_pdf(ctx, sample.dir_ws);
                if (sample.pdf <= 0.0)
                    return false;
                sample.radiance = g_scene.light_system.eval_environment_radiance(ctx, sample.dir_ws);
                sample.weight = sample.radiance / sample.pdf;
                selection_pdf = emitter_type_pmf.environment;
            }
        }

        // Adjust the conditional sample PDF and weight for the applicable selection probability.
        sample.pdf *= selection_pdf;
        sample.weight /= selection_pdf;

        // Propagate the emitter ID to the sample reference.
        sample_ref.emitter_id = sample.emitter_id;

        return true;
    }

    public float eval_analytic_selection_pdf(const LightSampleContext ctx, const LightID light_id)
    {
        if (uint(light_id) >= g_scene.light_system.analytic_light_count)
            return 0.0;
        const float probability
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power)
                  .analytic;
        return probability > 0.0 ? probability * analytic_light_distribution.eval_pdf(uint(light_id)) : 0.0;
    }

    float eval_environment_mixture_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        float pdf = 0.0;
        for (uint i = 0; i < g_scene.light_system.environment_light_count; ++i)
            pdf += environment_light_distribution.eval_pdf(i)
                * g_scene.light_system.get_environment_light(i).eval_pdf(ctx, dir_ws);
        return pdf;
    }

    public float eval_environment_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        const float environment_probability
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power)
                  .environment;
        if (environment_probability <= 0.0)
            return 0.0;
        return environment_probability * eval_environment_mixture_pdf(ctx, dir_ws);
    }

    public float eval_area_light_pdf(
        const LightSampleContext ctx,
        const LightID light_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        if (uint(light_id) >= g_scene.light_system.analytic_light_count)
            return 0.0;
        const float analytic_probability
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power)
                  .analytic;
        if (analytic_probability <= 0.0)
            return 0.0;
        return analytic_probability * analytic_light_distribution.eval_pdf(uint(light_id))
            * g_scene.light_system.get_light(light_id)
                  .eval_surface_sampling_pdf(ctx, emitter_pos_ws, emitter_normal_ws);
    }

    public float eval_emissive_pdf(
        const LightSampleContext ctx,
        const EmissiveTriangleID triangle_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        const float emissive_probability
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power)
                  .emissive_triangle;
        if (emissive_probability <= 0.0)
            return 0.0;
        const float triangle_selection_pdf = emissive_triangle_tree.eval_pdf(ctx, triangle_id);
        if (triangle_selection_pdf <= 0.0)
            return 0.0;
        return emissive_probability * triangle_selection_pdf
            * eval_emissive_triangle_pdf(ctx, triangle_id, emitter_pos_ws, emitter_normal_ws);
    }
}
