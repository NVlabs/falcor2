// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "geometry_vertex_attribute.h"
#include "falcor2/render/shared_geometry_types.h"

namespace falcor {
struct StaticMeshGeometryDataDesc;

struct MeshGeometryData {
    std::string name;
    std::vector<shared::PackedTriangleVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<MeshVertexAttribute> vertex_attributes;
};

// Replace output data, reusing vector capacity. Optional remaps identify the input vertex for each output.
void make_mesh_geometry_data(
    const StaticMeshGeometryDataDesc& desc,
    std::vector<MeshGeometryData>& sub_meshes,
    std::vector<std::vector<uint32_t>>* remaps = nullptr
);
void pack_mesh_vertices(
    std::span<const MeshVertexAttributeDesc> attributes,
    std::span<const uint32_t> remap,
    std::span<shared::PackedTriangleVertex> output
);
} // namespace falcor
