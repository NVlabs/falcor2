// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

#include "falcor2/testing/shader_test.slangh"

LightShaping make_light_shaping(bool enabled, float cutoff_angle, float softness, float focus)
{
    const float falloff_angle = cutoff_angle * (1.0 - softness);
    const float angular_falloff_width = radians(cutoff_angle - falloff_angle);
    const float angular_falloff_scale = angular_falloff_width > 0.0 ? 1.0 / angular_falloff_width : 0.0;

    LightShaping shaping;
    shaping.enabled = enabled;
    shaping.cos_cutoff_angle = cos(radians(cutoff_angle));
    shaping.cos_falloff_angle = cos(radians(falloff_angle));
    shaping.focus = focus;
    shaping.cutoff_angle = radians(cutoff_angle);
    shaping.angular_falloff_scale = angular_falloff_scale;
    return shaping;
}

float eval_at_angle(LightShaping shaping, float angle)
{
    const float angle_radians = radians(angle);
    return shaping.eval(float3(0.0, 0.0, 1.0), float3(sin(angle_radians), 0.0, cos(angle_radians)));
}

TEST_CASE(light_shaping_eval)
{
    const LightShaping disabled = make_light_shaping(false, 30.0, 0.5, 2.0);
    CHECK_APPROX(eval_at_angle(disabled, 180.0), 1.0, 1e-6);

    const LightShaping hard_cone = make_light_shaping(true, 30.0, 0.0, 0.0);
    CHECK_APPROX(eval_at_angle(hard_cone, 20.0), 1.0, 1e-6);
    CHECK_APPROX(eval_at_angle(hard_cone, 40.0), 0.0, 1e-6);

    const LightShaping soft_cone = make_light_shaping(true, 40.0, 0.25, 0.0);
    CHECK_APPROX(eval_at_angle(soft_cone, 25.0), 1.0, 1e-6);
    CHECK_APPROX(eval_at_angle(soft_cone, 35.0), 0.5, 1e-5);
    CHECK_APPROX(eval_at_angle(soft_cone, 45.0), 0.0, 1e-6);

    const LightShaping focused = make_light_shaping(true, 40.0, 0.25, 2.0);
    const float cos_20_degrees = cos(radians(20.0));
    const float cos_35_degrees = cos(radians(35.0));
    CHECK_APPROX(eval_at_angle(focused, 20.0), cos_20_degrees * cos_20_degrees, 1e-6);
    CHECK_APPROX(eval_at_angle(focused, 35.0), 0.5 * cos_35_degrees * cos_35_degrees, 1e-5);

    // Evaluate a very narrow transition without subtracting large, nearly equal affine-transform terms.
    // Near 90 degrees, this width spans only about 13 float32 angle steps. Backend trig rounding is amplified by the
    // inverse falloff width, so allow a wider factor tolerance while still rejecting a collapsed transition.
    const LightShaping narrow_soft_cone = make_light_shaping(true, 90.0, 1e-6, 0.0);
    const float narrow_angle = 90.0 * (1.0 - 0.25e-6);
    const float narrow_angle_radians = radians(narrow_angle);
    const float narrow_cos_theta = cos(narrow_angle_radians);
    const float narrow_delta
        = saturate((narrow_soft_cone.cutoff_angle - acos(narrow_cos_theta)) * narrow_soft_cone.angular_falloff_scale);
    CHECK(narrow_delta > 0.0 && narrow_delta < 0.5);
    CHECK_APPROX(eval_at_angle(narrow_soft_cone, narrow_angle), smoothstep(0.0, 1.0, narrow_delta), 2e-2);
}

TEST_CASE(point_light_sample)
{
    LightHeader header;
    PointLight light = PointLight(
        header,
        float3(0.0, 0.0, 2.0),
        float3(0.0, 0.0, -1.0),
        float3(2.0, 3.0, 4.0),
        make_light_shaping(false, 30.0, 0.5, 2.0)
    );

    LightSampleContext ctx = {};
    UniformSampleGenerator sg = UniformSampleGenerator(1);
    LightSample sample;
    CHECK(light.sample(ctx, next_float2(sg), sample));
    CHECK_APPROX(sample.dir_ws, float3(0.0, 0.0, 1.0), 1e-6);
    CHECK_APPROX(sample.distance, 2.0, 1e-6);
    CHECK_APPROX(sample.pdf, 1.0, 1e-6);
    CHECK_APPROX(sample.weight, float3(0.5, 0.75, 1.0), 1e-6);

    light.dir_ws = float3(1.0, 0.0, 0.0);
    light.shaping = make_light_shaping(true, 30.0, 0.0, 0.0);
    CHECK_FALSE(light.sample(ctx, next_float2(sg), sample));
}

TEST_CASE(analytic_area_light_sample)
{
    const float3 shading_pos_ws = float3(0.0);
    const float3 sampled_pos_ws = float3(0.0, 0.0, 2.0);
    const float3 sampled_normal_ws = float3(0.0, 0.0, -1.0);
    const float surface_area = 3.0;
    const float3 radiance = float3(2.0, 4.0, 6.0);
    const LightShaping unshaped = make_light_shaping(false, 30.0, 0.0, 0.0);

    LightSample sample;
    CHECK(finalize_analytic_area_light_sample(
        shading_pos_ws,
        sampled_pos_ws,
        sampled_normal_ws,
        surface_area,
        radiance,
        unshaped,
        sampled_normal_ws,
        sample
    ));
    CHECK_APPROX(sample.distance, 2.0, 1e-6);
    CHECK_APPROX(sample.pdf, 4.0 / 3.0, 1e-6);
    CHECK_APPROX(sample.weight, float3(1.5, 3.0, 4.5), 1e-6);

    CHECK_FALSE(finalize_analytic_area_light_sample(
        shading_pos_ws,
        sampled_pos_ws,
        -sampled_normal_ws,
        surface_area,
        radiance,
        unshaped,
        sampled_normal_ws,
        sample
    ));

    const LightShaping rejected = make_light_shaping(true, 30.0, 0.0, 0.0);
    CHECK_FALSE(finalize_analytic_area_light_sample(
        shading_pos_ws,
        sampled_pos_ws,
        sampled_normal_ws,
        surface_area,
        radiance,
        rejected,
        -sampled_normal_ws,
        sample
    ));
}

TEST_CASE(planar_lights_reject_backfacing_receiver_and_zero_area)
{
    LightSampleContext ctx = {};
    LightSample sample;

    LightHeader header;
    DiskLight disk = DiskLight(
        header,
        float3(0.0, 0.0, 1.0),
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0),
        float3(1.0),
        1.0,
        make_light_shaping(false, 30.0, 0.0, 0.0)
    );
    const float2 u_light = float2(0.375, 0.625);
    CHECK_FALSE(disk.sample(ctx, u_light, sample));
    CHECK_EQ(sample.pdf, 0.0);
    disk.normal_ws = float3(0.0, 0.0, -1.0);
    disk.surface_area = 0.0;
    CHECK_FALSE(disk.sample(ctx, u_light, sample));
    CHECK_EQ(sample.pdf, 0.0);

    RectLight rect = RectLight(
        header,
        float3(0.0, 0.0, 1.0),
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0),
        float3(1.0),
        1.0,
        make_light_shaping(false, 30.0, 0.0, 0.0)
    );
    CHECK_FALSE(rect.sample(ctx, u_light, sample));
    CHECK_EQ(sample.pdf, 0.0);
    rect.normal_ws = float3(0.0, 0.0, -1.0);
    rect.surface_area = 0.0;
    CHECK_FALSE(rect.sample(ctx, u_light, sample));
    CHECK_EQ(sample.pdf, 0.0);
}
