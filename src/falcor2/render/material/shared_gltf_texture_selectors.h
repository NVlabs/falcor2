// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/utils/math/bitfield.h"

#define FALCOR_SHARED_HOST
#define public

namespace falcor::shared {

using namespace sgl::math;

#include "falcor2/render/materials/gltf_texture_selectors.slang"

} // namespace falcor::shared

#undef public
#undef FALCOR_SHARED_HOST
