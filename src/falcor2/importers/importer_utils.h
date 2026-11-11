// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"

#include <cmath>
#include <limits>

namespace falcor {

/// Normalize value or fallback, using +Z if neither has a finite, normal (non-subnormal) squared length.
inline float3 normalize_or(const float3& value, const float3& fallback)
{
    const float length_squared = sgl::math::dot(value, value);
    if (std::isfinite(length_squared) && length_squared >= std::numeric_limits<float>::min())
        return value * (1.f / std::sqrt(length_squared));

    const float fallback_length_squared = sgl::math::dot(fallback, fallback);
    if (std::isfinite(fallback_length_squared) && fallback_length_squared >= std::numeric_limits<float>::min())
        return fallback * (1.f / std::sqrt(fallback_length_squared));
    return float3(0.f, 0.f, 1.f);
}

} // namespace falcor
