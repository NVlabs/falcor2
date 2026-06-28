// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TODO(scene): We should probably find a more well-defined way to share types between host and device code.

#include "falcor2/core/types.h"
#include "falcor2/core/enum.h"
#include "falcor2/utils/math/packing.h"

#define FALCOR_SHARED_HOST
#define public

namespace falcor::shared {

using namespace sgl::math;

#include "falcor2/render/scene_types.slang"
#include "falcor2/render/emissive_geometry_types.slang"
#include "falcor2/render/geometry/triangle_geometry_types.slang"
#include "falcor2/render/geometry/lss_geometry_types.slang"

} // namespace falcor::shared

#undef FALCOR_SHARED_HOST
#undef public
