// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/utils/camera_framing.h"
#include "falcor2/ui/scene_editor_gizmos.h"

#include <array>
#include <cmath>
#include <limits>

using namespace falcor;

static void check_fit(std::span<const AABB> bounds, const detail::CameraFraming& framing, float fov_y, float aspect)
{
    const auto view = sgl::math::inverse(framing.world_from_camera);
    CHECK_EQ(framing.world_from_camera.get_row(3), float4(0.f, 0.f, 0.f, 1.f));
    const float tan_y = std::tan(sgl::math::radians(fov_y) * 0.5f);
    CHECK_GT(framing.focus_distance, 0.f);
    CHECK_GT(framing.depth_range.x, 0.f);
    CHECK_GT(framing.depth_range.y, framing.depth_range.x);
    for (const auto& aabb : bounds) {
        for (uint32_t corner = 0; corner < 8; ++corner) {
            const float3 point(
                corner & 1 ? aabb.max.x : aabb.min.x,
                corner & 2 ? aabb.max.y : aabb.min.y,
                corner & 4 ? aabb.max.z : aabb.min.z
            );
            const float3 p = sgl::math::mul(view, float4(point, 1.f)).xyz();
            const float depth = -p.z;
            CHECK_GT(depth, framing.depth_range.x);
            CHECK_LT(depth, framing.depth_range.y);
            CHECK_LE(std::abs(p.x) / (depth * tan_y * aspect), 0.901f);
            CHECK_LE(std::abs(p.y) / (depth * tan_y), 0.901f);
        }
    }
}

TEST_SUITE_BEGIN("CameraFraming");

TEST_CASE("empty and invalid bounds keep the default view")
{
    const std::array<AABB, 2> bounds{AABB{}, AABB(float3(0.f), float3(std::numeric_limits<float>::infinity()))};
    const auto framing = detail::frame_camera(bounds, 50.f, 1.f);
    CHECK_EQ(framing.world_from_camera, sgl::math::matrix_from_translation(float3(0.f, 0.f, 5.f)));
    CHECK_EQ(framing.focus_distance, 5.f);
    CHECK_EQ(framing.depth_range, float2(0.01f, 1000.f));
    CHECK_EQ(detail::frame_camera(bounds, 50.f, 1.f, float2(0.5f, 200.f)).depth_range, float2(0.5f, 200.f));
    CHECK_THROWS(detail::frame_camera({}, 0.f, 1.f));
    CHECK_THROWS(detail::frame_camera({}, 180.f, 1.f));
    CHECK_THROWS(detail::frame_camera({}, 50.f, 0.f));
}

TEST_CASE("perspective framing fits transformed deep bounds and outliers")
{
    const auto world_from_object = sgl::math::mul(
        sgl::math::inverse(
            sgl::math::matrix_from_look_at(float3(30.f, -10.f, 15.f), float3(34.f, -9.f, 17.f), float3(0.f, 1.f, 0.f))
        ),
        sgl::math::matrix_from_scaling(float3(2.f, 3.f, 4.f))
    );
    const std::array<AABB, 3> bounds{
        AABB(float3(-2.f, -1.f, -20.f), float3(2.f, 1.f, 20.f)),
        AABB(float3(-1.f), float3(1.f)).transform(world_from_object),
        AABB(float3(100.f, 5.f, 8.f), float3(101.f, 6.f, 9.f)),
    };
    for (float aspect : {0.25f, 1.f, 4.f})
        for (float fov_y : {20.f, 90.f})
            check_fit(bounds, detail::frame_camera(bounds, fov_y, aspect), fov_y, aspect);
}

TEST_CASE("framing handles point bounds and planar geometry")
{
    const std::array<AABB, 1> point{AABB(float3(10.f, 20.f, 30.f), float3(10.f, 20.f, 30.f))};
    check_fit(point, detail::frame_camera(point, 50.f, 0.5f), 50.f, 0.5f);
    const std::array<AABB, 1> plane{AABB(float3(-10.f, -5.f, 0.f), float3(10.f, 5.f, 0.f))};
    check_fit(plane, detail::frame_camera(plane, 50.f, 2.f), 50.f, 2.f);
}

TEST_CASE("translating bounds preserves the chosen view")
{
    std::array<AABB, 2> bounds{AABB(float3(-2.f), float3(2.f)), AABB(float3(3.f), float3(4.f))};
    const auto before = detail::frame_camera(bounds, 40.f, 1.5f);
    const float3 offset(10.f, 20.f, 30.f);
    for (auto& aabb : bounds) {
        aabb.min += offset;
        aabb.max += offset;
    }
    const auto after = detail::frame_camera(bounds, 40.f, 1.5f);
    CHECK_LT(
        sgl::math::length(
            after.world_from_camera.get_col(3).xyz() - before.world_from_camera.get_col(3).xyz() - offset
        ),
        1e-4f
    );
    CHECK_EQ(after.focus_distance, doctest::Approx(before.focus_distance));
    check_fit(bounds, after, 40.f, 1.5f);
}

TEST_CASE("framed clipping planes leave room for navigation and nearby gizmos")
{
    const std::array<AABB, 1> plane{AABB(float3(-2.f, -2.f, 0.f), float3(2.f, 2.f, 0.f))};
    const auto framing = detail::frame_camera(plane, 50.f, 1.f);
    const float distance = framing.focus_distance;
    const auto projection
        = sgl::math::perspective(sgl::math::radians(50.f), 1.f, framing.depth_range.x, framing.depth_range.y);
    for (const float3 position : {
             framing.world_from_camera.get_col(3).xyz(),
             float3(0.f, 0.f, distance - 0.01f),
             float3(0.f, 0.f, distance + 1.f),
             float3(distance * 0.5f, 0.f, distance * 0.866f),
         }) {
        const auto view = sgl::math::matrix_from_look_at(position, float3(0.f), float3(0.f, 1.f, 0.f));
        // Camera/light guides in front of and behind the flat geometry must remain visible.
        for (float z : {-1.f, 0.f, 1.f}) {
            CHECK(
                ui::detail::project_gizmo_segment(
                    float3(-0.1f, 0.f, z),
                    float3(0.1f, 0.f, z),
                    sgl::math::mul(projection, view),
                    float2(0.f),
                    float2(100.f)
                )
                    .has_value()
            );
        }
    }
}

TEST_CASE("clipping adapts to tiny and large scenes without narrowing the baseline")
{
    for (float scale : {1e-4f, 1e4f}) {
        const std::array<AABB, 1> bounds{AABB(float3(-scale), float3(scale))};
        const auto framing = detail::frame_camera(bounds, 50.f, 1.f);
        check_fit(bounds, framing, 50.f, 1.f);
        CHECK_LE(framing.depth_range.x, 0.01f);
        CHECK_GE(framing.depth_range.y, 1000.f);
        if (scale < 1.f)
            CHECK_LT(framing.depth_range.x, 0.01f);
        else
            CHECK_GT(framing.depth_range.y, 1000.f);
    }
    const std::array<AABB, 1> bounds{AABB(float3(-1.f), float3(1.f))};
    const float2 baseline(0.001f, 20000.f);
    const auto framing = detail::frame_camera(bounds, 50.f, 1.f, baseline);
    CHECK_EQ(framing.depth_range, baseline);
    check_fit(bounds, framing, 50.f, 1.f);
}

TEST_SUITE_END();
