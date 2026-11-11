// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"

#include <sgl/core/data_type.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace falcor {

enum class MeshAttributeSemantic {
    position,
    normal,
    tangent,
    bitangent,
    handedness,
    tex_coord,
    color,
    joints,
    weights,
    unknown,
};
SGL_ENUM_INFO(
    MeshAttributeSemantic,
    {
        {MeshAttributeSemantic::position, "position"},
        {MeshAttributeSemantic::normal, "normal"},
        {MeshAttributeSemantic::tangent, "tangent"},
        {MeshAttributeSemantic::bitangent, "bitangent"},
        {MeshAttributeSemantic::handedness, "handedness"},
        {MeshAttributeSemantic::tex_coord, "tex_coord"},
        {MeshAttributeSemantic::color, "color"},
        {MeshAttributeSemantic::joints, "joints"},
        {MeshAttributeSemantic::weights, "weights"},
        {MeshAttributeSemantic::unknown, "unknown"},
    }
);
SGL_ENUM_REGISTER(MeshAttributeSemantic);

/// Description of an authored per-vertex geometry attribute.
struct GeometryVertexAttributeDesc {
    std::string_view name;
    MeshAttributeSemantic semantic{MeshAttributeSemantic::unknown};
    uint32_t index{0};
    sgl::DataType type{sgl::DataType::void_};
    uint32_t component_count{0};
    const void* data{nullptr};
    size_t stride{0};
    size_t data_size{0};
};

/// Owned, tightly packed per-vertex geometry attribute.
struct GeometryVertexAttribute {
    std::string name;
    MeshAttributeSemantic semantic{MeshAttributeSemantic::unknown};
    uint32_t index{0};
    sgl::DataType type{sgl::DataType::void_};
    uint32_t component_count{0};
    std::vector<uint8_t> data;
};

/// Return the size of one source scalar, accounting for the host representation of bool.
size_t get_geometry_attribute_type_size(sgl::DataType type);

/// Validate the memory layout and extent of an authored vertex attribute.
void validate_geometry_attribute_storage(
    const GeometryVertexAttributeDesc& attribute,
    uint32_t vertex_count,
    std::string_view geometry_type
);

/// Copy an authored vertex attribute into tightly packed storage using the supplied vertex remapping.
GeometryVertexAttribute
copy_geometry_vertex_attribute(const GeometryVertexAttributeDesc& source, std::span<const uint32_t> vertex_remap);

// Preserve the existing native mesh API names while sharing the representation with other geometry types.
using MeshVertexAttributeDesc = GeometryVertexAttributeDesc;
using MeshVertexAttribute = GeometryVertexAttribute;

} // namespace falcor
