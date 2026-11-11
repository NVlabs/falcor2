// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "static_mesh_geometry.h"

#include "geometry_attribute_registry.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/material.h"

#include <sgl/core/type_utils.h>

#include <cstring>

namespace falcor {

StaticMeshGeometry::~StaticMeshGeometry() { }

void StaticMeshGeometry::set_mesh_data(const StaticMeshGeometryDataDesc& desc)
{
    set_name(desc.name);

    make_mesh_geometry_data(desc, m_sub_meshes);
    if (m_scene)
        register_geometry_attributes();

    recompute_local_aabb();

    mark_dirty(DirtyFlags::render_state);
}

MeshVertexAttribute*
StaticMeshGeometry::find_vertex_attribute(size_t sub_mesh_index, MeshAttributeSemantic semantic, uint32_t index)
{
    for (MeshVertexAttribute& attribute : m_sub_meshes.at(sub_mesh_index).vertex_attributes) {
        if (attribute.semantic == semantic && attribute.index == index)
            return &attribute;
    }
    return nullptr;
}

const MeshVertexAttribute*
StaticMeshGeometry::find_vertex_attribute(size_t sub_mesh_index, MeshAttributeSemantic semantic, uint32_t index) const
{
    return const_cast<StaticMeshGeometry*>(this)->find_vertex_attribute(sub_mesh_index, semantic, index);
}

void StaticMeshGeometry::recompute_local_aabb()
{
    m_local_aabb = AABB();
    for (const SubMesh& sub_mesh : m_sub_meshes) {
        for (const shared::PackedTriangleVertex& vertex : sub_mesh.vertices) {
            m_local_aabb.expand(float3(vertex.position[0], vertex.position[1], vertex.position[2]));
        }
    }
}

void StaticMeshGeometry::on_add_to_scene()
{
    register_geometry_attributes();
}

void StaticMeshGeometry::register_geometry_attributes()
{
    for (const SubMesh& sub_mesh : m_sub_meshes) {
        for (const MeshVertexAttribute& attribute : sub_mesh.vertex_attributes)
            register_geometry_attribute(*m_scene, attribute);
    }
}

void StaticMeshGeometry::update_render_state()
{
    clear_render_state();
    RenderScene* render_scene = m_scene->_render_scene();
    std::vector<RenderHandle> geometries;
    for (const SubMesh& sub_mesh : m_sub_meshes) {
        RenderTriangleGeometryDesc desc = {};
        desc.vertex_count = sgl::narrow_cast<uint32_t>(sub_mesh.vertices.size());
        desc.index_count = sgl::narrow_cast<uint32_t>(sub_mesh.indices.size());
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
}

void StaticMeshGeometry::clear_render_state()
{
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

FALCOR_SCENE_REGISTER_GEOMETRY(StaticMeshGeometry);

} // namespace falcor
