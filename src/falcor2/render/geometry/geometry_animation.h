// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_types.h"
#include "falcor2/render/shared_geometry_types.h"

#include <span>

namespace falcor {

struct GeometryAnimationSelection {
    uint32_t source0 = 0;
    uint32_t source1 = 0;
    float source1_weight = 0.f;

    bool operator==(const GeometryAnimationSelection&) const = default;
};

void validate_geometry_samples(
    std::span<const float> times,
    size_t count,
    AnimationInterpolationMode interpolation,
    std::span<const uint32_t> sample_indices = {}
);

FALCOR_API GeometryAnimationSelection select_geometry_samples(
    double time,
    std::span<const float> times,
    AnimationInterpolationMode interpolation,
    std::span<const uint32_t> sample_indices = {}
);

/// Chooses how one geometry updates its current/previous vertex history. Call update() once per Scene update:
/// - First evaluation: initialize both buffers to the evaluated pose.
/// - Inputs changed: copy old current to previous, then evaluate the new current pose.
/// - First unchanged update after a change: copy current to previous to clear motion.
/// - Further unchanged updates: no work.
/// Reset to {} when the output buffers need initialization.
struct GeometryAnimationHistory {
    bool initialized = false;
    bool previous_matches_current = true;

    shared::GeometryHistoryUpdate update(bool changed)
    {
        if (!initialized) {
            initialized = true;
            previous_matches_current = true;
            return shared::GeometryHistoryUpdate::initialize;
        }
        if (changed) {
            previous_matches_current = false;
            return shared::GeometryHistoryUpdate::capture_current;
        }
        if (!previous_matches_current) {
            previous_matches_current = true;
            return shared::GeometryHistoryUpdate::catch_up;
        }
        return shared::GeometryHistoryUpdate::none;
    }
};

} // namespace falcor
