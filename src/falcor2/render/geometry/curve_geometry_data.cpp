// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "curve_geometry_data.h"
#include "static_curve_geometry.h"
#include "geometry_attribute_registry.h"
#include <numeric>
#include <cstring>

namespace falcor {
namespace {

void validate_curve_attribute(const GeometryVertexAttributeDesc& attribute, uint32_t vertex_count)
{
    validate_geometry_attribute_storage(attribute, vertex_count, "Curve");

    bool valid = true;
    if (attribute.semantic == MeshAttributeSemantic::tex_coord)
        valid = attribute.index < 8 && attribute.component_count >= 2 && attribute.component_count <= 4;
    else if (attribute.semantic == MeshAttributeSemantic::color)
        valid = attribute.index == 0 && attribute.component_count >= 3 && attribute.component_count <= 4;
    SGL_CHECK(valid, "Unsupported curve vertex attribute semantic, index, or component count");
}

} // namespace

template<typename T>
inline const T&
get_vertex_attribute(const StaticCurveGeometryDataDesc::VertexAttributeStream<T>& stream, uint32_t index)
{
    static const T zero{0};
    return stream.data
        ? *reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(stream.data) + index * stream.stride)
        : zero;
}

template<typename T>
inline const T& get_index(const StaticCurveGeometryDataDesc::IndexStream<T>& stream, uint32_t index)
{
    static const T zero{0};
    return stream.data
        ? *reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(stream.data) + index * stream.stride)
        : zero;
}

void make_curve_geometry_data(
    const StaticCurveGeometryDataDesc& desc,
    std::vector<shared::PackedLSSVertex>& vertices,
    std::vector<uint32_t>& indices,
    std::vector<GeometryVertexAttribute>& vertex_attributes,
    std::vector<uint32_t>* remap
)
{
    validate_unique_geometry_attributes(desc.vertex_attributes, "curve");
    for (const GeometryVertexAttributeDesc& attribute : desc.vertex_attributes) {
        validate_curve_attribute(attribute, desc.vertex_count);
        warn_if_unsupported_geometry_attribute(attribute, "Curve");
    }

    const GeometryVertexAttributeDesc* texcoord_attribute = nullptr;
    for (const auto& attribute : desc.vertex_attributes) {
        if (attribute.semantic == MeshAttributeSemantic::tex_coord && attribute.index == 0
            && attribute.type == sgl::DataType::float32)
            texcoord_attribute = &attribute;
    }

    vertices.resize(desc.vertex_count);
    for (uint32_t i = 0; i < desc.vertex_count; ++i) {
        shared::LSSVertex vertex = {};
        vertex.position = get_vertex_attribute(desc.position_stream, i);
        vertex.radius = get_vertex_attribute(desc.radius_stream, i);
        if (texcoord_attribute) {
            const size_t stride = texcoord_attribute->stride ? texcoord_attribute->stride
                                                             : texcoord_attribute->component_count * sizeof(float);
            std::memcpy(
                &vertex.texcoord,
                static_cast<const uint8_t*>(texcoord_attribute->data) + i * stride,
                sizeof(float2)
            );
        }
        vertices[i] = shared::detail::pack_lss_vertex(vertex);
    }

    indices.resize(desc.index_count);
    for (uint32_t i = 0; i < desc.index_count; ++i) {
        indices[i] = get_index(desc.index_stream, i);
    }

    std::vector<uint32_t> vertex_remap(desc.vertex_count);
    std::iota(vertex_remap.begin(), vertex_remap.end(), 0u);

    // Convert list-mode indices to successive-mode for GPU consumption and retain the source vertex of every output.
    if (desc.indexing_mode == CurveIndexingMode::list) {
        vertex_remap = StaticCurveGeometry::convert_list_to_successive_indices(indices, vertices);
    } else {
        for (uint32_t index : indices)
            SGL_CHECK(size_t(index) + 1 < vertices.size(), "Successive curve index is outside the vertex range");
    }

    vertex_attributes.clear();
    vertex_attributes.reserve(desc.vertex_attributes.size());
    for (const GeometryVertexAttributeDesc& attribute : desc.vertex_attributes) {
        vertex_attributes.push_back(copy_geometry_vertex_attribute(attribute, vertex_remap));
    }

    if (remap)
        remap->assign(vertex_remap.begin(), vertex_remap.end());
}

} // namespace falcor
