// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/types.h"

namespace falcor {

/// Minimum supported color temperature in Kelvin.
constexpr float MIN_COLOR_TEMPERATURE = 1000.f;

/// Maximum supported color temperature in Kelvin.
constexpr float MAX_COLOR_TEMPERATURE = 10000.f;

/// Convert a blackbody color temperature to a luminance-normalized scene-linear Rec.709/D65 RGB color.
/// 6500 K maps to neutral white.
/// Values outside the supported range are clamped.
FALCOR_API float3 color_temperature_to_rgb(float color_temperature);

} // namespace falcor
