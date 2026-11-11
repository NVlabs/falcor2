// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "static_curve_geometry.h"

#include "geometry_attribute_registry.h"

#include "falcor2/render/scene.h"

#include <sgl/core/type_utils.h>

#include <numeric>
#include <optional>
#include <cstring>

namespace falcor {

namespace {

void validate_attribute(const GeometryVertexAttributeDesc& attribute, uint32_t vertex_count)
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

StaticCurveGeometry::~StaticCurveGeometry() { }

void StaticCurveGeometry::set_curve_data(const StaticCurveGeometryDataDesc& desc)
{
    set_name(desc.name);

    validate_unique_geometry_attributes(desc.vertex_attributes, "curve");
    for (const GeometryVertexAttributeDesc& attribute : desc.vertex_attributes) {
        validate_attribute(attribute, desc.vertex_count);
        warn_if_unsupported_geometry_attribute(attribute, "Curve");
    }

    if (m_scene) {
        for (const GeometryVertexAttributeDesc& attribute : desc.vertex_attributes)
            register_geometry_attribute(*m_scene, attribute);
    }

    const GeometryVertexAttributeDesc* texcoord_attribute = nullptr;
    for (const auto& attribute : desc.vertex_attributes) {
        if (attribute.semantic == MeshAttributeSemantic::tex_coord && attribute.index == 0
            && attribute.type == sgl::DataType::float32)
            texcoord_attribute = &attribute;
    }

    m_vertices.resize(desc.vertex_count);
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
        m_vertices[i] = shared::detail::pack_lss_vertex(vertex);
    }

    m_indices.resize(desc.index_count);
    for (uint32_t i = 0; i < desc.index_count; ++i) {
        m_indices[i] = get_index(desc.index_stream, i);
    }

    std::vector<uint32_t> vertex_remap(desc.vertex_count);
    std::iota(vertex_remap.begin(), vertex_remap.end(), 0u);

    // Convert list-mode indices to successive-mode for GPU consumption and retain the source vertex of every output.
    if (desc.indexing_mode == CurveIndexingMode::list) {
        vertex_remap = convert_list_to_successive_indices(m_indices, m_vertices);
    } else {
        for (uint32_t index : m_indices)
            SGL_CHECK(size_t(index) + 1 < m_vertices.size(), "Successive curve index is outside the vertex range");
    }

    m_vertex_attributes.clear();
    m_vertex_attributes.reserve(desc.vertex_attributes.size());
    for (const GeometryVertexAttributeDesc& attribute : desc.vertex_attributes) {
        m_vertex_attributes.push_back(copy_geometry_vertex_attribute(attribute, vertex_remap));
    }

    // Compute local AABB from vertex positions.
    m_local_aabb = AABB();
    for (const shared::PackedLSSVertex& packed_vertex : m_vertices) {
        const shared::LSSVertex vertex = shared::detail::unpack_lss_vertex(packed_vertex);
        m_local_aabb.expand(vertex.position, vertex.radius);
    }

    mark_dirty(DirtyFlags::render_state);
}

void StaticCurveGeometry::clear_invalid_references() { }

void StaticCurveGeometry::on_add_to_scene()
{
    register_geometry_attributes();
}

void StaticCurveGeometry::register_geometry_attributes()
{
    for (const GeometryVertexAttribute& attribute : m_vertex_attributes)
        register_geometry_attribute(*m_scene, attribute);
}

void StaticCurveGeometry::update_render_state()
{
    clear_render_state();
    RenderScene* render_scene = m_scene->_render_scene();

    if (m_vertices.size() > 0 && m_indices.size() > 0) {
        RenderLSSGeometryDesc desc = {};
        desc.vertex_count = sgl::narrow_cast<uint32_t>(m_vertices.size());
        desc.index_count = sgl::narrow_cast<uint32_t>(m_indices.size());
        m_render_geometry = render_scene->create_lss_geometry(desc);
        RenderLSSGeometryData data = {};
        data.vertices = m_vertices;
        data.indices = m_indices;
        PreparedGeometryAttributes additional_attributes(m_vertex_attributes.size());
        for (const GeometryVertexAttribute& attribute : m_vertex_attributes) {
            // An ordinary float2 TEXCOORD_0 is already stored in PackedLSSVertex, as for triangles.
            if (attribute.semantic == MeshAttributeSemantic::tex_coord && attribute.index == 0
                && attribute.type == sgl::DataType::float32 && attribute.component_count == 2)
                continue;
            additional_attributes.append(*m_scene, attribute, m_vertices.size());
        }
        data.additional_attributes = additional_attributes.attributes();
        render_scene->update_lss_geometry(m_render_geometry, data);
    }

    if (m_render_geometry) {
        RenderGeometryGroupDesc desc = {};
        RenderHandle geometries[1] = {m_render_geometry};
        desc.geometries = geometries;
        m_render_geometry_group = render_scene->create_geometry_group(desc);
    }
}

void StaticCurveGeometry::clear_render_state()
{
    RenderScene* render_scene = m_scene->_render_scene();

    if (m_render_geometry_group) {
        render_scene->destroy_geometry_group(m_render_geometry_group);
        m_render_geometry_group = {};
    }

    if (m_render_geometry) {
        render_scene->destroy_lss_geometry(m_render_geometry);
        m_render_geometry = {};
    }
}

/// Convert list-mode indices (pairs per segment) to successive-mode indices (one per segment).
/// If any pair is non-consecutive (i.e. indices[2k+1] != indices[2k]+1), vertices are rearranged
/// so each segment's endpoints are adjacent.
std::vector<uint32_t> StaticCurveGeometry::convert_list_to_successive_indices(
    std::vector<uint32_t>& indices,
    std::vector<shared::PackedLSSVertex>& vertices
)
{
    SGL_CHECK(indices.size() % 2 == 0, "List-mode index buffer must have an even number of entries");

    for (uint32_t index : indices)
        SGL_CHECK(index < vertices.size(), "List-mode curve index is outside the vertex range");

    uint32_t segment_count = static_cast<uint32_t>(indices.size()) / 2;

    // Check if all pairs are already consecutive.
    bool all_consecutive = true;
    for (uint32_t i = 0; i < segment_count; ++i) {
        if (indices[2 * i + 1] != indices[2 * i] + 1) {
            all_consecutive = false;
            break;
        }
    }

    if (all_consecutive) {
        // Fast path: just take the first index of each pair.
        std::vector<uint32_t> new_indices(segment_count);
        for (uint32_t i = 0; i < segment_count; ++i) {
            new_indices[i] = indices[2 * i];
        }
        indices = std::move(new_indices);
        std::vector<uint32_t> vertex_remap(vertices.size());
        std::iota(vertex_remap.begin(), vertex_remap.end(), 0u);
        return vertex_remap;
    } else {
        // Slow path: rearrange vertices so each segment's endpoints are adjacent.
        std::vector<shared::PackedLSSVertex> new_vertices;
        new_vertices.reserve(segment_count * 2);
        std::vector<uint32_t> vertex_remap;
        vertex_remap.reserve(segment_count * 2);
        std::vector<uint32_t> new_indices(segment_count);
        std::optional<uint32_t> previous_source_index;

        for (uint32_t i = 0; i < segment_count; ++i) {
            uint32_t i0 = indices[2 * i];
            uint32_t i1 = indices[2 * i + 1];

            if (!previous_source_index || *previous_source_index != i0) {
                new_vertices.push_back(vertices[i0]);
                vertex_remap.push_back(i0);
            }
            new_indices[i] = static_cast<uint32_t>(new_vertices.size()) - 1;
            new_vertices.push_back(vertices[i1]);
            vertex_remap.push_back(i1);
            previous_source_index = i1;
        }

        vertices = std::move(new_vertices);
        indices = std::move(new_indices);
        return vertex_remap;
    }
}

FALCOR_SCENE_REGISTER_GEOMETRY(StaticCurveGeometry);

} // namespace falcor
