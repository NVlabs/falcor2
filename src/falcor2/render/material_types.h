// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"

namespace falcor {

/// Alpha modes for glTF-style materials.
enum class AlphaMode {
    /// The alpha value is ignored and the rendered output is fully opaque.
    opaque = 0,
    /// Alpha values below the cutoff are discarded.
    mask = 1,
    /// Alpha values are evaluated stochastically during intersection traversal.
    blend = 2,
};
SGL_ENUM_INFO(
    AlphaMode,
    {
        {AlphaMode::opaque, "opaque"},
        {AlphaMode::mask, "mask"},
        {AlphaMode::blend, "blend"},
    }
);
SGL_ENUM_REGISTER(AlphaMode);

} // namespace falcor
