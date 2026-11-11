// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/error.h"
#include <cstdint>

namespace falcor::streaming::detail {

// Shared by encoder and transport so the SPS and SDP advertise the same level.
// Use the session ceiling so adaptation never changes the negotiated level.
// Limits are for Baseline profile from H.264 table A-1.
inline uint32_t h264_level(uint32_t width, uint32_t height, uint32_t fps, uint32_t max_bitrate = 14000000)
{
    FALCOR_CHECK(
        width >= 16 && height >= 16 && width <= 3840 && height <= 2160 && width % 2 == 0 && height % 2 == 0,
        "Dimensions must be even, at least 16, and at most 3840x2160"
    );
    FALCOR_CHECK(fps > 0 && fps <= 60, "FPS must be between 1 and 60");
    FALCOR_CHECK(
        max_bitrate >= 100000 && max_bitrate <= 50000000,
        "Maximum bitrate must be between 100000 and 50000000 bits/s"
    );
    struct LevelLimits {
        uint32_t level;
        uint32_t frame_macroblocks;
        uint32_t macroblocks_per_second;
        uint32_t bitrate;
    };
    constexpr LevelLimits limits[] = {
        {31, 3600, 108000, 14000000},
        {32, 5120, 216000, 20000000},
        {40, 8192, 245760, 20000000},
        {41, 8192, 245760, 50000000},
        {42, 8704, 522240, 50000000},
        {50, 22080, 589824, 135000000},
        {51, 36864, 983040, 240000000},
        {52, 36864, 2073600, 240000000},
    };
    const uint32_t columns = (width + 15) / 16;
    const uint32_t rows = (height + 15) / 16;
    const uint32_t count = columns * rows;
    for (const auto& limit : limits) {
        if (max_bitrate <= limit.bitrate && count <= limit.frame_macroblocks
            && count * fps <= limit.macroblocks_per_second && columns * columns <= 8 * limit.frame_macroblocks
            && rows * rows <= 8 * limit.frame_macroblocks)
            return limit.level;
    }
    FALCOR_THROW("Video format exceeds H.264 level 5.2");
}

} // namespace falcor::streaming::detail
