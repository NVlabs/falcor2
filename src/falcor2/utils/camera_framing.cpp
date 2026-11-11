// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "camera_framing.h"
#include "falcor2/core/error.h"

#include <sgl/math/matrix_math.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace falcor::detail {
namespace {

float median(std::span<float> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5f;
}

} // namespace

CameraFraming frame_camera(std::span<const AABB> bounds, float fov_y, float aspect, float2 default_depth_range)
{
    FALCOR_CHECK(fov_y > 0.f && fov_y < 180.f, "Camera FOV must be between 0 and 180 degrees.");
    FALCOR_CHECK(std::isfinite(aspect) && aspect > 0.f, "Camera aspect ratio must be positive and finite.");
    FALCOR_CHECK(
        default_depth_range.x > 0.f && default_depth_range.y > default_depth_range.x
            && std::isfinite(default_depth_range.y),
        "Default camera depth range must be finite and satisfy 0 < near < far."
    );

    std::vector<float3> points;
    std::array<std::vector<float>, 3> centers;
    AABB scene_bounds;
    for (const AABB& aabb : bounds) {
        if (!aabb.is_valid() || !aabb.is_finite())
            continue;
        scene_bounds.expand(aabb);
        const float3 center = aabb.center();
        for (int axis = 0; axis < 3; ++axis)
            centers[axis].push_back(center[axis]);
        for (uint32_t corner = 0; corner < 8; ++corner)
            points.emplace_back(
                corner & 1 ? aabb.max.x : aabb.min.x,
                corner & 2 ? aabb.max.y : aabb.min.y,
                corner & 4 ? aabb.max.z : aabb.min.z
            );
    }

    CameraFraming result;
    result.depth_range = default_depth_range;
    if (points.empty()) {
        result.world_from_camera = sgl::math::matrix_from_translation(float3(0.f, 0.f, 5.f));
        return result;
    }

    const float3 target(median(centers[0]), median(centers[1]), median(centers[2]));
    const float tan_y = std::tan(sgl::math::radians(fov_y) * 0.5f);
    const float tan_x = tan_y * aspect;
    constexpr float COVERAGE = 0.9f;
    const float depth_margin = std::max(sgl::math::length(scene_bounds.size()) * 1e-4f, 1e-3f);
    const std::array<float3, 14> candidate_dirs = {
        float3(0.f, 0.f, -1.f),
        float3(0.f, 0.f, 1.f),
        float3(1.f, 0.f, 0.f),
        float3(-1.f, 0.f, 0.f),
        float3(0.707f, 0.f, -0.707f),
        float3(-0.707f, 0.f, -0.707f),
        float3(0.707f, 0.f, 0.707f),
        float3(-0.707f, 0.f, 0.707f),
        float3(0.5f, 0.5f, -0.707f),
        float3(-0.5f, 0.5f, -0.707f),
        float3(0.5f, -0.5f, -0.707f),
        float3(-0.5f, -0.5f, -0.707f),
        float3(0.f, 0.5f, -0.866f),
        float3(0.f, -0.5f, -0.866f),
    };
    float best_score = -std::numeric_limits<float>::infinity();
    for (const float3& candidate : candidate_dirs) {
        const float3 view_dir = sgl::math::normalize(candidate);
        const float3 up = std::abs(view_dir.y) > 0.98f ? float3(0.f, 0.f, 1.f) : float3(0.f, 1.f, 0.f);
        const float3 right = sgl::math::normalize(sgl::math::cross(view_dir, up));
        const float3 camera_up = sgl::math::cross(right, view_dir);
        float distance = 0.01f;
        float max_abs_x = 0.f;
        float max_abs_y = 0.f;
        float min_z = std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();
        for (const float3& point : points) {
            const float3 q = point - target;
            const float x = sgl::math::dot(q, right);
            const float y = sgl::math::dot(q, camera_up);
            const float z = sgl::math::dot(q, view_dir);
            max_abs_x = std::max(max_abs_x, std::abs(x));
            max_abs_y = std::max(max_abs_y, std::abs(y));
            // A corner's depth from the camera is distance + z, not just distance.
            distance = std::max(distance, std::abs(x) / (tan_x * COVERAGE) - z);
            distance = std::max(distance, std::abs(y) / (tan_y * COVERAGE) - z);
            distance = std::max(distance, depth_margin - z);
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
        }
        // Keep the importer's target-plane coverage score. Maximum projected corner fill
        // would be COVERAGE for every fitted view and would not distinguish their quality.
        const float fill = std::max(max_abs_x / (tan_x * distance), max_abs_y / (tan_y * distance));
        const float score = fill - 0.15f * (max_z - min_z) / distance;
        // Keep the first equivalent view so tiny roundoff changes do not flip the camera.
        if (score <= best_score + 1e-5f)
            continue;
        best_score = score;
        // Build the camera pose directly so its affine bottom row stays exact.
        result.world_from_camera = sgl::math::matrix_from_columns(
            float4(right, 0.f),
            float4(camera_up, 0.f),
            float4(-view_dir, 0.f),
            float4(target - view_dir * distance, 1.f)
        );
        result.focus_distance = distance;
        // Leave room to navigate and display editor gizmos outside the geometry bounds.
        const float near_plane = std::min(default_depth_range.x, (distance + min_z) * 0.1f);
        const float far_plane = std::max(default_depth_range.y, (distance + max_z) * 10.f);
        result.depth_range = float2(near_plane, far_plane);
    }
    return result;
}

} // namespace falcor::detail
