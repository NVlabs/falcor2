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
    CHECK_FALSE(sampler.sample_distance(float3(1.0), make_volume_distance_sample_input(sg), distance));
    CHECK_FALSE(isfinite(distance));
    CHECK_APPROX(sampler.eval_surface_weight(float3(1.0), 2.0), exp(-medium.sigma_a * 2.0), 1e-6f);
}

TEST_CASE(homogeneous_volume_sampler_samples_finite_scattering_distance)
{
    const let medium = make_medium(float3(0.1), float3(1.0));
    const let sampler = HomogeneousVolumeSampler(medium);
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(7);
    float distance;
    CHECK(sampler.sample_distance(float3(1.0), make_volume_distance_sample_input(sg), distance));
    CHECK(isfinite(distance));
    CHECK(distance >= 0.0);
    CHECK(all(isfinite(sampler.eval_event_weight(float3(1.0), distance))));
}

TEST_CASE(homogeneous_volume_sampler_explicit_channel_boundaries)
{
    const let sampler = HomogeneousVolumeSampler(make_medium(float3(0.0), float3(1.0, 2.0, 4.0)));
    const float3 throughput = float3(1.0, 1.0, 2.0);
    const float selectors[6] = { 0.0, 0x1.fffffep-3, 0.25, 0x1.fffffep-2, 0.5, 0x1.fffffep-1 };
    const float rates[6] = { 1.0, 1.0, 2.0, 2.0, 4.0, 4.0 };
    for (uint i = 0; i < 6; ++i) {
        const VolumeDistanceSampleInput input = { selectors[i], 0.5 };
        float distance, replay;
        CHECK(sampler.sample_distance(throughput, input, distance));
        CHECK(sampler.sample_distance(throughput, input, replay));
        CHECK_EQ(distance, replay);
        CHECK_APPROX(distance, log(2.0) / rates[i], 1.0e-6);
    }
}

TEST_CASE(homogeneous_volume_sampler_explicit_distance_endpoints)
{
    const let sampler = HomogeneousVolumeSampler(make_medium(float3(0.5), float3(2.0)));
    float distance;
    CHECK(sampler.sample_distance(float3(1.0), VolumeDistanceSampleInput(0.0, 0.0), distance));
    CHECK_EQ(distance, 0.0);
    CHECK(sampler.sample_distance(float3(1.0), VolumeDistanceSampleInput(0.5, 0x1.fffffep-1), distance));
    CHECK(isfinite(distance));
    CHECK_APPROX(distance, 12.0 * log(2.0), 1.0e-5);
}

TEST_CASE(homogeneous_volume_sampler_zero_channel_probability_and_rate)
{
    const let sampler = HomogeneousVolumeSampler(make_medium(float3(0.0), float3(0.0, 2.0, 0.0)));
    float distance;
    // Zero proposal probabilities are skipped even at exact coordinate boundaries.
    CHECK(sampler.sample_distance(float3(0.0, 1.0, 0.0), VolumeDistanceSampleInput(0.0, 0.5), distance));
    CHECK_APPROX(distance, log(2.0) / 2.0, 1.0e-6);
    CHECK(sampler.sample_distance(float3(0.0, 1.0, 0.0), VolumeDistanceSampleInput(0x1.fffffep-1, 0.5), distance));
    CHECK_APPROX(distance, log(2.0) / 2.0, 1.0e-6);
    // A zero rate with positive proposal mass is a successful escape sample.
    CHECK(sampler.sample_distance(float3(1.0), VolumeDistanceSampleInput(0.0, 0.0), distance));
    CHECK(distance == float.INFINITY);
    CHECK(sampler.sample_distance(float3(1.0), VolumeDistanceSampleInput(0x1.fffffep-1, 0.5), distance));
    CHECK(distance == float.INFINITY);

    const let absorbing = HomogeneousVolumeSampler(make_medium(float3(1.0), float3(0.0)));
    CHECK_FALSE(absorbing.sample_distance(float3(1.0), VolumeDistanceSampleInput(0.5, 0.5), distance));
    CHECK(distance == float.INFINITY);
}
