// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry_vertex_attribute.h"

#include <sgl/core/error.h>

#include <cstring>
#include <limits>

namespace falcor {

size_t get_geometry_attribute_type_size(sgl::DataType type)
{
    return type == sgl::DataType::bool_ ? sizeof(bool) : sgl::data_type_size(type);
}

void validate_geometry_attribute_storage(
    const GeometryVertexAttributeDesc& attribute,
    uint32_t vertex_count,
    std::string_view geometry_type
)
{
    SGL_CHECK(attribute.data, "{} vertex attribute data must not be null", geometry_type);
    SGL_CHECK(attribute.type != sgl::DataType::void_, "{} vertex attributes must have a data type", geometry_type);
    SGL_CHECK(attribute.component_count > 0, "{} vertex attributes must have at least one component", geometry_type);

    const size_t element_size = attribute.component_count * get_geometry_attribute_type_size(attribute.type);
    SGL_CHECK(
        attribute.stride >= element_size,
        "{} vertex attribute stride is smaller than its element size",
        geometry_type
    );
    if (vertex_count > 0) {
        SGL_CHECK(
            attribute.stride <= (std::numeric_limits<size_t>::max() - element_size) / (vertex_count - 1),
            "{} vertex attribute size overflows",
            geometry_type
        );
        const size_t required_size = (vertex_count - 1) * attribute.stride + element_size;
        SGL_CHECK(
            attribute.data_size >= required_size,
            "{} vertex attribute data is smaller than its vertex range",
            geometry_type
        );
    }
}

GeometryVertexAttribute
copy_geometry_vertex_attribute(const GeometryVertexAttributeDesc& source, std::span<const uint32_t> vertex_remap)
{
    GeometryVertexAttribute result{
        .name = std::string(source.name),
        .semantic = source.semantic,
        .index = source.index,
        .type = source.type,
        .component_count = source.component_count,
    };

    const size_t element_size = source.component_count * get_geometry_attribute_type_size(source.type);
    result.data.resize(vertex_remap.size() * element_size);
    for (size_t vertex_index = 0; vertex_index < vertex_remap.size(); ++vertex_index) {
        std::memcpy(
            result.data.data() + vertex_index * element_size,
            reinterpret_cast<const uint8_t*>(source.data) + vertex_remap[vertex_index] * source.stride,
            element_size
        );
    }
    return result;
}

} // namespace falcor
