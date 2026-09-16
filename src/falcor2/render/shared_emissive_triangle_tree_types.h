// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"
#include "falcor2/core/types.h"
#include "falcor2/utils/math/packing.h"

#include <cmath>

#define FALCOR_SHARED_HOST
#define public

namespace falcor::shared {

using namespace sgl::math;

#include "falcor2/render/emissive_triangle_tree_types.slang"

} // namespace falcor::shared

#undef public
#undef FALCOR_SHARED_HOST

static_assert(sizeof(falcor::shared::PackedEmissiveTriangleTreeNode) == 32);
static_assert(sizeof(falcor::shared::EmissiveTriangleTreeNode) == 56);
