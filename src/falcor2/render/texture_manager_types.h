// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/core/enum.h"
#include "falcor2/utils/math/packing.h"
#include "falcor2/utils/math/bitfield.h"

#define FALCOR_SHARED_HOST
#define public

namespace falcor::shared {

using namespace sgl::math;

#include "falcor2/render/texture_manager_types.slang"

} // namespace falcor::shared

#undef FALCOR_SHARED_HOST
#undef public
