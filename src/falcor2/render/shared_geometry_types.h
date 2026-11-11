// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/shared_scene_types.h"
#include "falcor2/utils/buffer_ptr.h"

#define FALCOR_SHARED_HOST
#define public

namespace falcor::shared {

using namespace sgl::math;

#include "falcor2/render/geometry/area_light_geometry_types.slang"
#include "falcor2/render/geometry/triangle_geometry_types.slang"
#include "falcor2/render/geometry/lss_geometry_types.slang"
#include "falcor2/render/geometry/geometry_animation_types.slang"

} // namespace falcor::shared

#undef FALCOR_SHARED_HOST
#undef public
