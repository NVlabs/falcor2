// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/render/material/gltf_specialization.h"
#include "falcor2/render/scene_types.h"

namespace falcor {

/// Immutable configuration of a live scene.
struct FALCOR_API SceneConfig {
    /// Target image origin convention for scene texture coordinates.
    ///
    /// Individual ImporterMesh instances still carry their authored texcoord
    /// convention and are converted to this target convention while loading.
    UVOrigin uv_origin = UVOrigin::upper_left;

    /// Texture size limit as a full mip count (13 allows 4096 for power-of-two textures); zero imposes no limit.
    /// USD root-layer customLayerData falcor:maxMipCount can tighten this at scene creation; the smaller nonzero limit
    /// wins. Applies to subsequent loads and appends, including embedded images, but not already-created GPU textures.
    /// Appended assets do not change the cap. DDS without a fitting mip uses its smallest available mip and warns.
    uint32_t max_mip_count{0};

    /// Select how native glTF materials are specialized. Default of balanced builds a small number
    /// of specializations to cover all requirements.
    GLTFSpecializationMode gltf_specialization_mode{GLTFSpecializationMode::balanced};

    /// Hard feature cap for native glTF materials.
    GLTFFeatureFlags gltf_allowed_features{GLTFFeatureFlags::all};
};

} // namespace falcor
