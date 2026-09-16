// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/importer_types.h"
#include "falcor2/core/macros.h"

namespace falcor {

/// Immutable configuration of a live scene.
struct FALCOR_API SceneConfig {
    /// Target image origin convention for scene texture coordinates.
    ///
    /// Individual ImporterMesh instances still carry their authored texcoord
    /// convention and are converted to this target convention while loading.
    UVOrigin uv_origin = UVOrigin::upper_left;
};

} // namespace falcor
