// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/core/macros.h"

#include <limits>

namespace falcor {

/// Axis-Aligned Bounding Box.
struct FALCOR_API AABB {
    /// Minimum point of the AABB.
    float3 min = float3(std::numeric_limits<float>::max());
    /// Maximum point of the AABB.
    float3 max = float3(std::numeric_limits<float>::lowest());

    /// Default constructor. Creates an invalid AABB.
    AABB() = default;

    /// Constructor with explicit min/max points.
    AABB(float3 min_pt, float3 max_pt)
        : min(min_pt)
        , max(max_pt)
    {
    }

    /// Check if the AABB is valid (min <= max for all components).
    bool is_valid() const { return min.x <= max.x && min.y <= max.y && min.z <= max.z; }

    /// Expand the AABB to include a point.
    void expand(const float3& point)
    {
        if (!is_valid()) {
            min = max = point;
        } else {
            min = sgl::math::min(min, point);
            max = sgl::math::max(max, point);
        }
    }

    /// Expand the AABB to include a sphere (point + radius).
    void expand(const float3& point, float radius)
    {
        float3 r{radius};
        expand(point - r);
        expand(point + r);
    }

    /// Expand the AABB to include another AABB.
    void expand(const AABB& other)
    {
        if (!other.is_valid())
            SGL_THROW("Cannot expand using an invalid AABB");
        if (!is_valid()) {
            *this = other;
        } else {
            min = sgl::math::min(min, other.min);
            max = sgl::math::max(max, other.max);
        }
    }

    /// Transform the AABB by a matrix (creates new AABB containing all 8 corners).
    AABB transform(const float4x4& matrix) const
    {
        if (!is_valid())
            SGL_THROW("Cannot transform an invalid AABB");

        AABB result;
        // Transform all 8 corners of the AABB
        for (int i = 0; i < 8; ++i) {
            float3 corner = float3((i & 1) ? max.x : min.x, (i & 2) ? max.y : min.y, (i & 4) ? max.z : min.z);
            float4 transformed = sgl::math::mul(matrix, float4(corner, 1.0f));
            result.expand(float3(transformed.x, transformed.y, transformed.z));
        }
        return result;
    }

    /// Get the center of the AABB.
    float3 center() const
    {
        if (!is_valid())
            SGL_THROW("Invalid AABB");
        return (min + max) * 0.5f;
    }

    /// Get the size (extent) of the AABB.
    float3 size() const
    {
        if (!is_valid())
            SGL_THROW("Invalid AABB");
        return (max - min);
    }
};

} // namespace falcor
