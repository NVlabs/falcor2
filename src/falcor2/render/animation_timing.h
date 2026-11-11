// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"

#include <algorithm>
#include <limits>

namespace falcor {

/// Animation interval in seconds, allowing negative or nonzero starts.
/// Default construction produces an empty interval; a single time (start == end) is valid.
struct FALCOR_API AnimationTimeRange {
    float start = std::numeric_limits<float>::max();
    float end = std::numeric_limits<float>::lowest();

    bool is_valid() const { return start <= end; }

    /// Expand to include another interval, which must be valid or default constructed.
    void expand(const AnimationTimeRange& other)
    {
        start = std::min(start, other.start);
        end = std::max(end, other.end);
    }
};

} // namespace falcor
