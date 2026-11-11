// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/utils/aabb.h"

#include <span>

namespace falcor::detail {

struct CameraFraming {
    float4x4 world_from_camera = float4x4::identity();
    float focus_distance{5.f};
    float2 depth_range{0.01f, 1000.f};
};

/// Choose a view that fits all finite world-space bounds, using a median instance center.
/// Without valid bounds, look at the origin from (0,0,5).
/// @param bounds World-space bounds to frame. Invalid or non-finite bounds are ignored.
/// @param fov_y Vertical field of view in degrees, strictly between 0 and 180.
/// @param aspect Viewport width/height ratio. Must be positive and finite.
/// @param default_depth_range Finite clipping range satisfying 0 < near < far. Framing may widen it
/// for scene scale and navigation.
FALCOR_API CameraFraming frame_camera(
    std::span<const AABB> bounds,
    float fov_y,
    float aspect = 1.f,
    float2 default_depth_range = float2(0.01f, 1000.f)
);

} // namespace falcor::detail
