// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/shared_scene_types.h"

#define FALCOR_SHARED_HOST
#define public

namespace falcor::shared {

using namespace sgl::math;

#include "falcor2/render/emissive_geometry_types.slang"

} // namespace falcor::shared

#undef FALCOR_SHARED_HOST
#undef public
