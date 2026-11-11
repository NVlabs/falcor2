// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "geometry_vertex_attribute.h"
#include "falcor2/render/shared_geometry_types.h"

namespace falcor {
struct StaticCurveGeometryDataDesc;

// Replace output data. Vector capacity is reused where the indexing conversion permits it.
// The optional remap identifies the input vertex for each output, so all samples use the same ordering.
void make_curve_geometry_data(
    const StaticCurveGeometryDataDesc& desc,
    std::vector<shared::PackedLSSVertex>& vertices,
    std::vector<uint32_t>& indices,
    std::vector<GeometryVertexAttribute>& vertex_attributes,
    std::vector<uint32_t>* remap = nullptr
);
} // namespace falcor
