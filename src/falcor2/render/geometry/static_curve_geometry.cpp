// SPDX-License-Identifier: Apache-2.0

#include "static_curve_geometry.h"

#include "falcor2/render/scene.h"

#include <sgl/core/type_utils.h>

#include <cstring>

namespace falcor {

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

    m_vertices.resize(desc.vertex_count);
    for (uint32_t i = 0; i < desc.vertex_count; ++i) {
        shared::LSSVertex vertex = {};
        vertex.position = get_vertex_attribute(desc.position_stream, i);
        vertex.radius = get_vertex_attribute(desc.radius_stream, i);
        m_vertices[i] = shared::detail::pack_lss_vertex(vertex);
    }

    m_indices.resize(desc.index_count);
    for (uint32_t i = 0; i < desc.index_count; ++i) {
        m_indices[i] = get_index(desc.index_stream, i);
    }

    // Convert list-mode indices to successive-mode for GPU consumption.
    if (desc.indexing_mode == CurveIndexingMode::list) {
        convert_list_to_successive_indices(m_indices, m_vertices);
    }

    // Compute local AABB from vertex positions.
    m_local_aabb = AABB();
    for (uint32_t i = 0; i < desc.vertex_count; ++i)
        m_local_aabb.expand(get_vertex_attribute(desc.position_stream, i), get_vertex_attribute(desc.radius_stream, i));

    mark_dirty(DirtyFlags::render_state);
}

void StaticCurveGeometry::clear_invalid_references() { }

void StaticCurveGeometry::update_render_state()
{
    RenderScene* render_scene = m_scene->_render_scene();

    if (m_render_geometry_group) {
        render_scene->destroy_geometry_group(m_render_geometry_group);
        m_render_geometry_group = {};
    }

    if (m_render_geometry) {
        render_scene->destroy_lss_geometry(m_render_geometry);
    }

    if (m_vertices.size() > 0 && m_indices.size() > 0) {
        RenderLSSGeometryDesc desc = {};
        desc.vertex_count = sgl::narrow_cast<uint32_t>(m_vertices.size());
        desc.index_count = sgl::narrow_cast<uint32_t>(m_indices.size());
        m_render_geometry = render_scene->create_lss_geometry(desc);
        RenderLSSGeometryData data = {};
        data.vertices = m_vertices;
        data.indices = m_indices;
        render_scene->update_lss_geometry(m_render_geometry, data);
    }

    if (m_render_geometry) {
        RenderGeometryGroupDesc desc = {};
        RenderHandle geometries[1] = {m_render_geometry};
        desc.geometries = geometries;
        m_render_geometry_group = render_scene->create_geometry_group(desc);
    }
}

/// Convert list-mode indices (pairs per segment) to successive-mode indices (one per segment).
/// If any pair is non-consecutive (i.e. indices[2k+1] != indices[2k]+1), vertices are rearranged
/// so each segment's endpoints are adjacent.
void StaticCurveGeometry::convert_list_to_successive_indices(
    std::vector<uint32_t>& indices,
    std::vector<shared::PackedLSSVertex>& vertices
)
{
    SGL_CHECK(indices.size() % 2 == 0, "List-mode index buffer must have an even number of entries");

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
    } else {
        // Slow path: rearrange vertices so each segment's endpoints are adjacent.
        std::vector<shared::PackedLSSVertex> new_vertices;
        new_vertices.reserve(segment_count + 1);
        std::vector<uint32_t> new_indices(segment_count);

        for (uint32_t i = 0; i < segment_count; ++i) {
            uint32_t i0 = indices[2 * i];
            uint32_t i1 = indices[2 * i + 1];

            if (i == 0 || std::memcmp(&new_vertices.back(), &vertices[i0], sizeof(shared::PackedLSSVertex)) != 0) {
                new_vertices.push_back(vertices[i0]);
            }
            new_indices[i] = static_cast<uint32_t>(new_vertices.size()) - 1;
            new_vertices.push_back(vertices[i1]);
        }

        vertices = std::move(new_vertices);
        indices = std::move(new_indices);
    }
}

FALCOR_SCENE_REGISTER_GEOMETRY(StaticCurveGeometry);

} // namespace falcor
