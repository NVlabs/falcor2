// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dynamic_curve_geometry.h"
#include "geometry_attribute_registry.h"
#include "falcor2/render/scene.h"
#include "falcor2/utils/buffer_ptr.h"

#include <sgl/core/type_utils.h>
#include <sgl/core/short_vector.h>

namespace falcor {

void DynamicCurveGeometry::set_curve_data(
    std::vector<shared::PackedLSSVertex> vertices,
    std::vector<uint32_t> indices,
    std::vector<GeometryVertexAttribute> vertex_attributes,
    std::unique_ptr<GeometryDeformer<shared::PackedLSSVertex>> deformer
)
{
    FALCOR_CHECK(m_scene, "Dynamic geometry must belong to a Scene");
    FALCOR_ASSERT(deformer);
    if (m_deformer)
        m_deformer->detach();
    m_vertices = std::move(vertices);
    m_indices = std::move(indices);
    m_vertex_attributes = std::move(vertex_attributes);
    m_deformer = std::move(deformer);
    m_local_aabb = m_deformer->local_aabb();
    register_geometry_attributes();
    mark_dirty(DirtyFlags::render_state);
}

void DynamicCurveGeometry::on_add_to_scene()
{
    register_geometry_attributes();
}

void DynamicCurveGeometry::on_remove_from_scene()
{
    clear_render_state();
}

void DynamicCurveGeometry::register_geometry_attributes()
{
    for (const GeometryVertexAttribute& attribute : m_vertex_attributes)
        register_geometry_attribute(*m_scene, attribute);
}

void DynamicCurveGeometry::update_render_state()
{
    clear_render_state();
    RenderScene* render_scene = m_scene->_render_scene();

    if (m_vertices.size() > 0 && m_indices.size() > 0) {
        RenderLSSGeometryDesc desc = {};
        desc.vertex_count = sgl::narrow_cast<uint32_t>(m_vertices.size());
        desc.index_count = sgl::narrow_cast<uint32_t>(m_indices.size());
        desc.is_dynamic = true;
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
    if (m_deformer && m_render_geometry)
        m_deformer->attach(
            *render_scene,
            [this](
                shared::GeometryHistoryUpdate history_update,
                sgl::short_vector<GeometryDeformationTarget, 1>& targets
            )
            {
                prepare_output(history_update, targets);
            }
        );
}

void DynamicCurveGeometry::clear_render_state()
{
    if (m_deformer)
        m_deformer->detach();
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

void DynamicCurveGeometry::prepare_output(
    shared::GeometryHistoryUpdate history_update,
    sgl::short_vector<GeometryDeformationTarget, 1>& targets
)
{
    m_local_aabb = m_deformer->local_aabb();
    auto* render_scene = m_scene->_render_scene();
    auto handle = m_render_geometry;
    const auto& output = std::as_const(*render_scene).get_lss_geometry(handle);
    targets.push_back({
        .current = to_rw_buffer_ptr(
            render_scene->lss_vertex_buffer(handle, 0),
            sizeof(shared::PackedLSSVertex) * output.vertex_offset
        ),
        .previous = to_rw_buffer_ptr(
            render_scene->lss_vertex_buffer(handle, 1),
            sizeof(shared::PackedLSSVertex) * output.vertex_offset
        ),
        .vertex_count = output.vertex_count,
    });
    if (history_update != shared::GeometryHistoryUpdate::catch_up) {
        render_scene->mark_lss_geometry_deformed(handle);
        mark_dirty(DirtyFlags::deformation);
    }
}

FALCOR_SCENE_REGISTER_GEOMETRY(DynamicCurveGeometry);

} // namespace falcor
