// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/error.h"
#include "falcor2/core/types.h"

#include <sgl/device/device.h>
#include <sgl/device/kernel.h>
#include <sgl/device/pipeline.h>

#include <algorithm>
#include <cstdint>

namespace falcor {

/// Maximum tile dimensions in threads for kernels with one geometry per row (group Y/Z must be one).
inline uint2 geometry_dispatch_max_tile_size(uint3 group_size, uint3 group_limit)
{
    FALCOR_ASSERT(group_size.x > 0 && group_size.y == 1 && group_size.z == 1);
    FALCOR_ASSERT(group_limit.x > 0 && group_limit.y > 0);
    // SGL rounds thread counts up using (count + group_size - 1) in uint32_t.
    const uint32_t safe_threads = UINT32_MAX - (group_size.x - 1);
    const uint32_t groups_x = std::min(group_limit.x, safe_threads / group_size.x);
    FALCOR_ASSERT(groups_x > 0);
    return {groups_x * group_size.x, group_limit.y};
}

inline uint2 geometry_dispatch_max_tile_size(sgl::ComputeKernel* kernel)
{
    return geometry_dispatch_max_tile_size(
        kernel->pipeline()->thread_group_size(),
        kernel->device()->info().limits.max_compute_dispatch_thread_groups
    );
}

} // namespace falcor
