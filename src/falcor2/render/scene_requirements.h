// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/fwd.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"

#include <sgl/device/shader.h>

#include <vector>

namespace falcor {

/// Scene requirements for the shader compilation.
struct FALCOR_API SceneRequirements {
    std::vector<ref<sgl::SlangModule>> modules;
    std::vector<sgl::TypeConformance> type_conformances;

    /// Required ray-tracing pipeline flags.
    sgl::RayTracingPipelineFlags ray_tracing_pipeline_flags = sgl::RayTracingPipelineFlags::none;

    auto operator<=>(const SceneRequirements&) const = default;
};

} // namespace falcor
