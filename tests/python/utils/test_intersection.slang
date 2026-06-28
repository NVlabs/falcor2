// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/testing/shader_test.slangh"
import falcor2.utils;

// ----------------------------------------------------------------------------
// intersect_ray_sphere
// ----------------------------------------------------------------------------

TEST_CASE(intersect_ray_sphere_hit)
{
    float t;
    bool hit = intersect_ray_sphere(float3(0, 0, -5), float3(0, 0, 1), float3(0), 1.0, t);
    CHECK(hit);
    CHECK_APPROX(t, 4.0f, 1e-6f);
}

TEST_CASE(intersect_ray_sphere_miss)
{
    float t;
    bool hit = intersect_ray_sphere(float3(0, 0, -5), float3(1, 0, 0), float3(0), 1.0, t);
    CHECK_FALSE(hit);
}

// ----------------------------------------------------------------------------
// intersect_ray_aabb
// ----------------------------------------------------------------------------

TEST_CASE(intersect_ray_aabb_hit)
{
    float2 near_far;
    bool hit = intersect_ray_aabb(float3(0, 0, -5), float3(0, 0, 1), float3(-1), float3(1), near_far);
    CHECK(hit);
    CHECK_APPROX(near_far.x, 4.0f, 1e-6f);
    CHECK_APPROX(near_far.y, 6.0f, 1e-6f);
}

TEST_CASE(intersect_ray_aabb_miss)
{
    float2 near_far;
    bool hit = intersect_ray_aabb(float3(3, 0, -5), float3(0, 0, 1), float3(-1), float3(1), near_far);
    CHECK_FALSE(hit);
}

// ----------------------------------------------------------------------------
// intersect_ray_triangle
// ----------------------------------------------------------------------------

TEST_CASE(intersect_ray_triangle_hit)
{
    float3 vertices[3] = {
        float3(0, 0, 0),
        float3(1, 0, 0),
        float3(0, 1, 0),
    };
    float3 barycentrics;
    float t;
    bool hit = intersect_ray_triangle(float3(0.25, 0.25, -1), float3(0, 0, 1), vertices, barycentrics, t);
    CHECK(hit);
    CHECK_APPROX(t, 1.0f, 1e-6f);
    // Barycentrics should be non-negative and sum to 1.
    CHECK(barycentrics.x >= -1e-6f);
    CHECK(barycentrics.y >= -1e-6f);
    CHECK(barycentrics.z >= -1e-6f);
    CHECK_APPROX(barycentrics.x + barycentrics.y + barycentrics.z, 1.0f, 1e-6f);
}

TEST_CASE(intersect_ray_triangle_miss)
{
    float3 vertices[3] = {
        float3(0, 0, 0),
        float3(1, 0, 0),
        float3(0, 1, 0),
    };
    float3 barycentrics;
    float t;
    bool hit = intersect_ray_triangle(float3(1.2, 1.2, -1), float3(0, 0, 1), vertices, barycentrics, t);
    CHECK_FALSE(hit);
}

// ----------------------------------------------------------------------------
// intersect_ray_linear_swept_sphere
// ----------------------------------------------------------------------------

TEST_CASE(intersect_ray_linear_swept_sphere_hit)
{
    float2 result;
    bool hit = intersect_ray_linear_swept_sphere_han19(
        float3(0, 0, 0),
        float3(0, 0, -1),
        float4(-0.5, 0, -3, 0.4),
        float4(0.5, 0, -3, 0.6),
        true,
        result
    );
    CHECK(hit);
    CHECK(result.x >= 2.0f);
    CHECK(result.x <= 3.5f);
    CHECK(result.y >= 0.0f);
    CHECK(result.y <= 1.0f);
}

TEST_CASE(intersect_ray_linear_swept_sphere_miss)
{
    float2 result;
    bool hit = intersect_ray_linear_swept_sphere_han19(
        float3(3, 0, 0),
        float3(0, 0, -1),
        float4(-0.5, 0, -3, 0.4),
        float4(0.5, 0, -3, 0.6),
        true,
        result
    );
    CHECK_FALSE(hit);
}
