// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.utils;

#include "falcor2/testing/shader_test.slangh"

void check_frame(float3 normal, float3 tangent, float handedness, Frame.Component component)
{
    const Frame frame = Frame(normal, tangent, handedness, component);
    CHECK_APPROX(length(frame.normal), 1.0, 1.0e-6);
    CHECK_APPROX(length(frame.tangent), 1.0, 1.0e-6);
    CHECK_APPROX(length(frame.bitangent), 1.0, 1.0e-6);
    CHECK_APPROX(dot(frame.normal, frame.tangent), 0.0, 1.0e-6);
    CHECK_APPROX(dot(frame.normal, frame.bitangent), 0.0, 1.0e-6);
    CHECK_APPROX(dot(frame.tangent, frame.bitangent), 0.0, 1.0e-6);
    CHECK_APPROX(cross(frame.normal, frame.tangent) * handedness, frame.bitangent, 1.0e-6);
    if (component == Frame.Component::normal)
        CHECK_EQ(frame.normal, normal);
    else
        CHECK_EQ(frame.tangent, tangent);
    const float3 direction = normalize(float3(0.2, -0.3, 0.4));
    CHECK_APPROX(frame.to_global(frame.to_local(direction)), direction, 1.0e-6);
}

TEST_CASE(collinear_frame)
{
    for (int sign = -1; sign <= 1; sign += 2) {
        for (int handedness = -1; handedness <= 1; handedness += 2) {
            for (int component = 0; component < 2; ++component) {
                check_frame(float3(-1, 0, 0), float3(sign, 0, 0), handedness, Frame.Component(component));
                const float3 normal = normalize(float3(1, 2, 3));
                check_frame(normal, sign * normal, handedness, Frame.Component(component));
                check_frame(
                    float3(0, 0, 1),
                    normalize(float3(1.0e-7, 0, sign)),
                    handedness,
                    Frame.Component(component)
                );
            }
        }
    }
}

TEST_CASE(nondegenerate_frame_preserved)
{
    const Frame near_frame = Frame(float3(0, 0, 1), normalize(float3(2.0e-6, 0, 1)));
    CHECK_EQ(near_frame.normal, float3(0, 0, 1));
    CHECK_EQ(near_frame.tangent, float3(1, 0, 0));
    CHECK_EQ(near_frame.bitangent, float3(0, 1, 0));
    const float3 normal = normalize(float3(1, 2, 3));
    const float3 tangent = normalize(float3(3, 1, 2));
    const float3 bitangent = normalize(cross(normal, tangent));
    for (int handedness = -1; handedness <= 1; handedness += 2) {
        const Frame normal_frame = Frame(normal, tangent, handedness);
        CHECK_EQ(normal_frame.normal, normal);
        CHECK_EQ(normal_frame.tangent, cross(bitangent, normal));
        CHECK_EQ(normal_frame.bitangent, handedness * bitangent);
        const Frame tangent_frame = Frame(normal, tangent, handedness, Frame.Component::tangent);
        CHECK_EQ(tangent_frame.tangent, tangent);
        CHECK_EQ(tangent_frame.normal, cross(tangent, bitangent));
        CHECK_EQ(tangent_frame.bitangent, handedness * bitangent);
    }
}

[Differentiable]
float3 frame_direction(float3 normal, no_diff float3 tangent)
{
    return Frame(normalize(normal), tangent).to_global(float3(0.2, 0.3, 0.4));
}

TEST_CASE(frame_forward_derivative)
{
    const float3 normal = float3(0, 0, 1);
    const float3 derivative = float3(1, 0, 0);
    for (int collinear = 0; collinear < 2; ++collinear) {
        const float3 tangent = collinear != 0 ? normal : float3(0, 1, 0);
        let result = fwd_diff(frame_direction)(diffPair(normal, derivative), tangent);
        CHECK_FALSE(any(isnan(result.p)) || any(isinf(result.p)));
        CHECK_FALSE(any(isnan(result.d)) || any(isinf(result.d)));
        CHECK(length(result.d) > 0.0);
    }
}
