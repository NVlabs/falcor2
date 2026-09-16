// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/testing/shader_test.slangh"

import falcor2.render;
import falcor2.utils;

HomogeneousMediumProperties make_medium(float3 sigma_a, float3 sigma_s)
{
    HomogeneousMediumProperties medium = {};
    medium.sigma_a = sigma_a;
    medium.sigma_s = sigma_s;
    return medium;
}

TEST_CASE(homogeneous_volume_sampler_reports_scattering)
{
    CHECK_FALSE(HomogeneousVolumeSampler(make_medium(float3(1.0), float3(0.0))).has_scattering());
    CHECK(HomogeneousVolumeSampler(make_medium(float3(0.0), float3(0.0, 0.5, 0.0))).has_scattering());
}

TEST_CASE(homogeneous_volume_sampler_computes_channel_probabilities)
{
    const let medium = make_medium(float3(1.0, 1.0, 0.0), float3(1.0, 3.0, 0.0));
    const let sampler = HomogeneousVolumeSampler(medium);
    const float3 probabilities = sampler.channel_probabilities(float3(2.0, 1.0, 0.5));
    const float3 expected_weights = float3(1.0, 0.75, 0.5);
    CHECK_APPROX(
        probabilities,
        expected_weights / (expected_weights.x + expected_weights.y + expected_weights.z),
        1e-6f
    );

    const float3 fallback = sampler.channel_probabilities(float3(0.0));
    CHECK_APPROX(fallback, float3(1.0 / 3.0), 1e-6f);
}

TEST_CASE(homogeneous_volume_sampler_matches_achromatic_implicit_absorption)
{
    const let medium = make_medium(float3(0.25), float3(0.75));
    const let sampler = HomogeneousVolumeSampler(medium);
    const float distance = 2.0;
    const float3 expected = float3(exp(-0.25 * distance));
    CHECK_APPROX(sampler.eval_event_weight(float3(1.0), distance), expected, 1e-6f);
    CHECK_APPROX(sampler.eval_surface_weight(float3(1.0), distance), expected, 1e-6f);
}

TEST_CASE(homogeneous_volume_sampler_evaluates_extinction_transmittance)
{
    const let sampler = HomogeneousVolumeSampler(make_medium(float3(0.1, 0.2, 0.3), float3(0.4, 0.3, 0.2)));
    CHECK_APPROX(sampler.eval_transmittance(2.0), exp(float3(-1.0)), 1e-6f);
    CHECK_EQ(sampler.eval_transmittance(0.0), float3(1.0));

    const let partially_clear = HomogeneousVolumeSampler(make_medium(float3(0.0, 0.2, 0.0), float3(0.0, 0.0, 0.3)));
    CHECK_EQ(partially_clear.eval_transmittance_to_infinity(), float3(1.0, 0.0, 0.0));
    CHECK_EQ(partially_clear.eval_transmittance(float.INFINITY), float3(1.0, 0.0, 0.0));
}

TEST_CASE(homogeneous_volume_sampler_weights_escape_probability)
{
    const let sampler = HomogeneousVolumeSampler(make_medium(float3(0.0), float3(0.0, 1.0, 1.0)));
    CHECK_APPROX(sampler.eval_surface_weight_to_infinity(float3(1.0)), float3(3.0, 0.0, 0.0), 1e-6);
}

TEST_CASE(homogeneous_volume_sampler_keeps_pure_absorption_deterministic)
{
    const let medium = make_medium(float3(0.25, 0.5, 1.0), float3(0.0));
    const let sampler = HomogeneousVolumeSampler(medium);
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(1);
    float distance;
    CHECK_FALSE(sampler.sample_distance(float3(1.0), sg, distance));
    CHECK_FALSE(isfinite(distance));
    CHECK_APPROX(sampler.eval_surface_weight(float3(1.0), 2.0), exp(-medium.sigma_a * 2.0), 1e-6f);
}

TEST_CASE(homogeneous_volume_sampler_samples_finite_scattering_distance)
{
    const let medium = make_medium(float3(0.1), float3(1.0));
    const let sampler = HomogeneousVolumeSampler(medium);
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(7);
    float distance;
    CHECK(sampler.sample_distance(float3(1.0), sg, distance));
    CHECK(isfinite(distance));
    CHECK(distance >= 0.0);
    CHECK(all(isfinite(sampler.eval_event_weight(float3(1.0), distance))));
}
