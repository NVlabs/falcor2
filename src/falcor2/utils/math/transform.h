// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace falcor {

/// Return the volume-equivalent uniform scale of an affine transform.
inline float volume_equivalent_uniform_scale(const float4x4& transform)
{
    const float3 x = transform.get_col(0).xyz();
    const float3 y = transform.get_col(1).xyz();
    const float3 z = transform.get_col(2).xyz();
    return std::cbrt(std::abs(dot(x, cross(y, z))));
}

/// Return whether the linear part of an affine transform preserves spheres up to uniform scaling.
inline bool is_similarity_transform(const float4x4& transform, float relative_tolerance = 1e-5f)
{
    const float3 x = transform.get_col(0).xyz();
    const float3 y = transform.get_col(1).xyz();
    const float3 z = transform.get_col(2).xyz();
    const float x_squared = dot(x, x);
    const float y_squared = dot(y, y);
    const float z_squared = dot(z, z);
    const float max_squared = std::max({x_squared, y_squared, z_squared});
    if (!std::isfinite(max_squared))
        return false;
    if (max_squared == 0.f)
        return true;

    const float tolerance = relative_tolerance * max_squared;
    return std::abs(x_squared - y_squared) <= tolerance && std::abs(x_squared - z_squared) <= tolerance
        && std::abs(dot(x, y)) <= tolerance && std::abs(dot(x, z)) <= tolerance && std::abs(dot(y, z)) <= tolerance;
}

/// Return whether the linear part of an affine transform preserves axial symmetry around the given local axis.
/// The axial scale may differ from the equal radial scales, but all transformed axes must be non-degenerate and
/// orthogonal.
inline bool
is_axially_symmetric_transform(const float4x4& transform, uint32_t symmetry_axis, float relative_tolerance = 1e-4f)
{
    if (symmetry_axis >= 3)
        return false;

    const std::array axes = {
        transform.get_col(0).xyz(),
        transform.get_col(1).xyz(),
        transform.get_col(2).xyz(),
    };
    const std::array lengths = {
        length(axes[0]),
        length(axes[1]),
        length(axes[2]),
    };
    if (!std::isfinite(lengths[0]) || !std::isfinite(lengths[1]) || !std::isfinite(lengths[2]) || lengths[0] <= 0.f
        || lengths[1] <= 0.f || lengths[2] <= 0.f)
        return false;

    for (uint32_t i = 0; i < 3; ++i) {
        for (uint32_t j = i + 1; j < 3; ++j) {
            if (std::abs(dot(axes[i], axes[j])) > relative_tolerance * lengths[i] * lengths[j])
                return false;
        }
    }

    const uint32_t radial_axis0 = (symmetry_axis + 1) % 3;
    const uint32_t radial_axis1 = (symmetry_axis + 2) % 3;
    return std::abs(lengths[radial_axis0] - lengths[radial_axis1])
        <= relative_tolerance * std::max(lengths[radial_axis0], lengths[radial_axis1]);
}

} // namespace falcor
