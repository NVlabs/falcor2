// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry_animation.h"

#include <algorithm>
#include <cmath>
#include <sgl/core/type_utils.h>

namespace falcor {

void validate_geometry_samples(
    std::span<const float> times,
    size_t count,
    AnimationInterpolationMode interpolation,
    std::span<const uint32_t> sample_indices
)
{
    FALCOR_CHECK(!times.empty() && count > 0, "Motion requires samples and timestamps");
    FALCOR_CHECK(
        sample_indices.empty() ? times.size() == count : times.size() == sample_indices.size(),
        "Motion requires one sample index per timestamp"
    );
    for (uint32_t sample : sample_indices)
        FALCOR_CHECK(sample < count, "Motion sample index out of bounds");
    FALCOR_CHECK(
        interpolation == AnimationInterpolationMode::linear || interpolation == AnimationInterpolationMode::step,
        "Vertex caches support step or linear interpolation"
    );
    for (size_t i = 0; i < times.size(); ++i)
        FALCOR_CHECK(
            std::isfinite(times[i]) && (i == 0 || times[i] >= times[i - 1]) && (i < 2 || times[i] > times[i - 2]),
            "Motion times must be finite and increasing, with at most two keys at a jump"
        );
}

GeometryAnimationSelection select_geometry_samples(
    double time,
    std::span<const float> times,
    AnimationInterpolationMode interpolation,
    std::span<const uint32_t> sample_indices
)
{
    FALCOR_ASSERT(!times.empty());
    auto sample = [&](size_t key)
    {
        return sample_indices.empty() ? sgl::narrow_cast<uint32_t>(key) : sample_indices[key];
    };
    if (!std::isfinite(time) || time < times.front())
        return {sample(0), sample(0), 0.f};
    if (time >= times.back())
        return {sample(times.size() - 1), sample(times.size() - 1), 0.f};
    // upper_bound chooses the right-hand value at a jump; the preceding interval ends at its left-hand value.
    auto right = std::upper_bound(times.begin(), times.end(), time);
    size_t first = right - times.begin() - 1;
    uint32_t source0 = sample(first);
    uint32_t source1 = sample(first + 1);
    if (interpolation == AnimationInterpolationMode::step || time == times[first] || source0 == source1)
        return {source0, source0, 0.f};
    return {source0, source1, static_cast<float>((time - times[first]) / (times[first + 1] - times[first]))};
}

} // namespace falcor
