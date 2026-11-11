// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dynamic_mesh_geometry.h"
#include "geometry_attribute_registry.h"
#include "falcor2/render/scene.h"
#include "falcor2/utils/buffer_ptr.h"

#include <sgl/core/type_utils.h>
#include <sgl/core/short_vector.h>

namespace falcor {

void DynamicMeshGeometry::set_mesh_data(
    std::vector<MeshGeometryData> sub_meshes,
    std::unique_ptr<GeometryDeformer<shared::PackedTriangleVertex>> deformer
)
{
    FALCOR_CHECK(m_scene, "Dynamic geometry must belong to a Scene");
    FALCOR_ASSERT(deformer);
    if (m_deformer)
        m_deformer->detach();
    m_sub_meshes = std::move(sub_meshes);
    m_deformer = std::move(deformer);
    m_local_aabb = m_deformer->local_aabb();
    register_geometry_attributes();
    mark_dirty(DirtyFlags::render_state);
}

void DynamicMeshGeometry::on_add_to_scene()
{
    register_geometry_attributes();
}

void DynamicMeshGeometry::on_remove_from_scene()
{
    clear_render_state();
}

void DynamicMeshGeometry::register_geometry_attributes()
{
    for (const SubMesh& sub_mesh : m_sub_meshes) {
        for (const MeshVertexAttribute& attribute : sub_mesh.vertex_attributes)
            register_geometry_attribute(*m_scene, attribute);
    }
}

void DynamicMeshGeometry::update_render_state()
{
    clear_render_state();
    RenderScene* render_scene = m_scene->_render_scene();
    std::vector<RenderHandle> geometries;
    for (const SubMesh& sub_mesh : m_sub_meshes) {
        RenderTriangleGeometryDesc desc = {};
        desc.vertex_count = sgl::narrow_cast<uint32_t>(sub_mesh.vertices.size());
        desc.index_count = sgl::narrow_cast<uint32_t>(sub_mesh.indices.size());
        desc.is_dynamic = true;
        RenderTriangleGeometryHandle geometry = render_scene->create_triangle_geometry(desc);
        RenderTriangleGeometryData data = {};
        data.vertices = sub_mesh.vertices;
        data.indices = sub_mesh.indices;
        PreparedGeometryAttributes additional_attributes(sub_mesh.vertex_attributes.size());
        for (const MeshVertexAttribute& attribute : sub_mesh.vertex_attributes) {
            // An ordinary float2 TEXCOORD_0 is already stored in PackedTriangleVertex.
            if (attribute.semantic == MeshAttributeSemantic::tex_coord && attribute.index == 0
                && attribute.type == sgl::DataType::float32 && attribute.component_count == 2)
                continue;
            additional_attributes.append(*m_scene, attribute, sub_mesh.vertices.size());
        }
        data.additional_attributes = additional_attributes.attributes();
        render_scene->update_triangle_geometry(geometry, data);
        m_render_geometries.push_back(geometry);
        geometries.push_back(geometry);
    }

    if (geometries.size() > 0) {
        RenderGeometryGroupDesc desc = {};
        desc.geometries = geometries;
        m_render_geometry_group = render_scene->create_geometry_group(desc);
    }
    if (m_deformer && !m_render_geometries.empty())
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

void DynamicMeshGeometry::clear_render_state()
{
    if (m_deformer)
        m_deformer->detach();
    RenderScene* render_scene = m_scene->_render_scene();

    if (m_render_geometry_group) {
        render_scene->destroy_geometry_group(m_render_geometry_group);
        m_render_geometry_group = {};
    }

    for (RenderTriangleGeometryHandle geometry : m_render_geometries) {
        render_scene->destroy_triangle_geometry(geometry);
    }

    m_render_geometries.clear();
}

void DynamicMeshGeometry::prepare_output(
    shared::GeometryHistoryUpdate history_update,
    sgl::short_vector<GeometryDeformationTarget, 1>& targets
)
{
    m_local_aabb = m_deformer->local_aabb();
    auto* render_scene = m_scene->_render_scene();
    for (size_t i = 0; i < m_render_geometries.size(); ++i) {
        auto handle = m_render_geometries[i];
        const auto& output = std::as_const(*render_scene).get_triangle_geometry(handle);
        targets.push_back({
            .current = to_rw_buffer_ptr(
                render_scene->triangle_vertex_buffer(handle, 0),
                sizeof(shared::PackedTriangleVertex) * output.vertex_offset
            ),
            .previous = to_rw_buffer_ptr(
                render_scene->triangle_vertex_buffer(handle, 1),
                sizeof(shared::PackedTriangleVertex) * output.vertex_offset
            ),
            .vertex_count = output.vertex_count,
        });
        if (history_update != shared::GeometryHistoryUpdate::catch_up)
            render_scene->mark_triangle_geometry_deformed(handle);
    }
    if (history_update != shared::GeometryHistoryUpdate::catch_up)
        mark_dirty(DirtyFlags::deformation);
}

FALCOR_SCENE_REGISTER_GEOMETRY(DynamicMeshGeometry);

} // namespace falcor
