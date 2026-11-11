// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_capabilities.h"

#include <string>
#include <vector>

namespace falcor {

/// Consumer demand, independent of capability selection and output storage.
/// AOV names refer to existing scene declarations; duplicates and ordering do not affect resolution.
struct FALCOR_API SceneRenderRequest {
    std::vector<std::string> material_aovs;
    /// Consumer shader operations required independently of material evaluation and requested outputs.
    SceneCapabilities required_capabilities;
};

} // namespace falcor
