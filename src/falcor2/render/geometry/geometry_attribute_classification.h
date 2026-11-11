// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/geometry/geometry_vertex_attribute.h"

#include <cstdint>

namespace falcor {

enum class GeometryAttributeClassificationKind {
    core,
    fixed,
    named,
};

struct GeometryAttributeClassification {
    GeometryAttributeClassificationKind kind;
    uint32_t fixed_id{0};
};

/// Classify how a geometry vertex attribute is represented in the scene attribute registry.
GeometryAttributeClassification classify_geometry_attribute(MeshAttributeSemantic semantic, uint32_t index);

} // namespace falcor
