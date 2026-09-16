// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.utils;

#include "falcor2/testing/shader_test.slangh"

void check_adjusted_normal(
    const float3 shading_normal,
    const float3 geometric_normal,
    const float3 view_direction,
    const float handedness
)
{
    const float3 adjusted = adjust_shading_normal(shading_normal, geometric_normal, view_direction);
    Frame frame = Frame(shading_normal, float3(0.0, 1.0, 0.0), handedness);
    adjust_shading_normal(frame, geometric_normal, view_direction);

    CHECK_NE(adjusted, shading_normal);
    CHECK_FALSE(any(isnan(adjusted)));
    CHECK_FALSE(any(isinf(adjusted)));
    CHECK_APPROX(length(adjusted), 1.0, 1.0e-5);
    CHECK_EQ(dot(adjusted, geometric_normal) >= 0.0, dot(shading_normal, geometric_normal) >= 0.0);
    CHECK_APPROX(frame.normal, adjusted, 1.0e-6);
    CHECK_APPROX(dot(frame.normal, frame.tangent), 0.0, 1.0e-6);
    CHECK_APPROX(dot(frame.normal, frame.bitangent), 0.0, 1.0e-6);
    CHECK_EQ(frame.handedness_sign(), handedness);
}

TEST_CASE(shading_normal_adjustment_contract)
{
    const float3 front_normal = float3(0.0, 0.0, 1.0);
    Frame unchanged_frame = Frame(front_normal, normalize(float3(0.6, 0.8, 0.0)), -1.0);
    const Frame original_frame = unchanged_frame;
    CHECK_EQ(adjust_shading_normal(front_normal, front_normal, front_normal), front_normal);
    adjust_shading_normal(unchanged_frame, front_normal, front_normal);
    CHECK_EQ(unchanged_frame.normal, original_frame.normal);
    CHECK_EQ(unchanged_frame.tangent, original_frame.tangent);
    CHECK_EQ(unchanged_frame.bitangent, original_frame.bitangent);

    check_adjusted_normal(normalize(float3(1.0, 0.0, 0.05)), front_normal, front_normal, -1.0);

    const float3 back_normal = -front_normal;
    check_adjusted_normal(normalize(float3(1.0, 0.0, -0.05)), back_normal, back_normal, 1.0);
}
