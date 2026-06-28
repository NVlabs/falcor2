// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing rendering;

/// Sampler for scene light sources, including emissive triangles and environment map.
public struct SceneLightSampler<let ENABLE_EMISSIVE_TRIANGLES = true, let ENABLE_ENV_MAP = true>
{
    public UniformEmissiveTriangleSampler emissive_triangle_sampler;
    public EnvMapSampler env_map_sampler;

    static const uint EMITTER_TYPE_COUNT = 2;
    typealias TypeProbabilities = vector<float, EMITTER_TYPE_COUNT>;

    // [Differentiable, PreferRecompute, __NoSideEffect]
    public bool sample(
        no_diff const Interaction interaction,
        no_diff inout UniformSampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    )
    {
        sample = {};
        weight = {};

        EmitterType type = {};
        float type_pdf = {};
        if (!sample_type(sg, type, type_pdf))
            return false;

        if (ENABLE_EMISSIVE_TRIANGLES && type == EmitterType::emissive_triangle) {
            if (emissive_triangle_sampler.sample(interaction, sg, sample, weight)) {
                sample.pdf *= type_pdf;
                weight /= type_pdf;
                return true;
            }
        } else if (ENABLE_ENV_MAP && type == EmitterType::env_map) {
            if (env_map_sampler.sample(sg, sample, weight)) {
                sample.pdf *= type_pdf;
                weight /= type_pdf;
                return true;
            }
        }

        return false;
    }

    public float eval_pdf(const Interaction interaction, const DirectionSample sample)
    {
        float type_pdf = eval_type_pdf(sample.emitter_type);

        if (ENABLE_EMISSIVE_TRIANGLES && sample.emitter_type == EmitterType::emissive_triangle) {
            return emissive_triangle_sampler.eval_pdf(interaction, sample) * type_pdf;
        } else if (ENABLE_ENV_MAP && sample.emitter_type == EmitterType::env_map) {
            return env_map_sampler.eval_pdf(sample) * type_pdf;
        }
        return 0.0;
    }

    TypeProbabilities compute_type_propabilities()
    {
        float emissive_triangle = (ENABLE_EMISSIVE_TRIANGLES && g_scene.has_emissive_triangles()) ? 1.0 : 0.0;
        float env_map = (ENABLE_ENV_MAP && g_scene.has_env_map()) ? 1.0 : 0.0;
        float total = emissive_triangle + env_map;
        if (total > 0.0) {
            emissive_triangle /= total;
            env_map /= total;
        }
        return TypeProbabilities(emissive_triangle, env_map);
    }

    bool sample_type(inout UniformSampleGenerator sg, out EmitterType type, out float pdf)
    {
        TypeProbabilities p = compute_type_propabilities();
        float x = next_float(sg);
        if (ENABLE_EMISSIVE_TRIANGLES && x < p[0]) {
            type = EmitterType::emissive_triangle;
            pdf = p[0];
            return true;
        } else if (ENABLE_ENV_MAP && x < p[0] + p[1]) {
            type = EmitterType::env_map;
            pdf = p[1];
            return true;
        } else {
            type = EmitterType::unknown;
            pdf = 0.0;
            return false;
        }
    }

    float eval_type_pdf(EmitterType type)
    {
        TypeProbabilities p = compute_type_propabilities();
        if (ENABLE_EMISSIVE_TRIANGLES && type == EmitterType::emissive_triangle) {
            return p[0];
        } else if (ENABLE_ENV_MAP && type == EmitterType::env_map) {
            return p[1];
        } else {
            return 0.0;
        }
    }
}
