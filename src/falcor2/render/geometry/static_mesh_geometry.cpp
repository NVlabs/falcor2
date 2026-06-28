// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "static_mesh_geometry.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/material.h"

#include <sgl/core/type_utils.h>

namespace falcor {

template<typename T>
inline const T& get_vertex_attribute(const StaticMeshGeometryDataDesc::VertexAttributeStream<T>& stream, uint32_t index)
{
    static const T zero{0};
    return stream.data
        ? *reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(stream.data) + index * stream.stride)
        : zero;
}

template<typename T>
inline const T& get_index(const StaticMeshGeometryDataDesc::IndexStream<T>& stream, uint32_t index)
{
    static const T zero{0};
    return stream.data
        ? *reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(stream.data) + index * stream.stride)
        : zero;
}

StaticMeshGeometry::~StaticMeshGeometry() { }

void StaticMeshGeometry::set_mesh_data(const StaticMeshGeometryDataDesc& desc)
{
    set_name(desc.name);

    m_sub_meshes.resize(desc.sub_meshes.size());

    // Convert a single sub-mesh from the input data.
    // Extracts the required vertices, packs the data and remaps indices.
    // The output sub-mesh is self-contained.
    auto convert_sub_mesh = [&](size_t sub_mesh_index)
    {
        const StaticMeshGeometryDataDesc::SubMesh& src = desc.sub_meshes[sub_mesh_index];
        SubMesh& dst = m_sub_meshes[sub_mesh_index];

        dst.name = src.name;
        dst.material = nullptr;

        dst.indices.resize(src.index_count);
        uint32_t vertex_count = 0;
        std::vector<uint32_t> index_map(desc.vertex_count, 0xffffffff);
        std::vector<uint32_t> reverse_index_map(desc.vertex_count);
        for (uint32_t i = 0; i < src.index_count; ++i) {
            uint32_t orig_index = get_index(src.index_stream, i);
            if (index_map[orig_index] == 0xffffffff) {
                uint32_t new_index = vertex_count++;
                index_map[orig_index] = new_index;
                reverse_index_map[new_index] = orig_index;
            }
            dst.indices[i] = index_map[orig_index];
        }

        dst.vertices.resize(vertex_count);
        for (uint32_t i = 0; i < vertex_count; ++i) {
            shared::TriangleVertex vertex = {};
            uint32_t orig_index = reverse_index_map[i];
            vertex.position = get_vertex_attribute(desc.position_stream, orig_index);
            vertex.normal = get_vertex_attribute(desc.normal_stream, orig_index);
            vertex.tangent = get_vertex_attribute(desc.tangent_stream, orig_index);
            vertex.handedness = get_vertex_attribute(desc.handedness_stream, orig_index);
            vertex.uv[0] = get_vertex_attribute(desc.texcoord_stream, orig_index);
            dst.vertices[i] = shared::detail::pack_triangle_vertex(vertex);
        }
    };

    for (size_t sub_mesh_index = 0; sub_mesh_index < desc.sub_meshes.size(); ++sub_mesh_index)
        convert_sub_mesh(sub_mesh_index);

    recompute_local_aabb();

    mark_dirty(DirtyFlags::render_state);
}

void StaticMeshGeometry::clear_invalid_references() { }

void StaticMeshGeometry::recompute_local_aabb()
{
    m_local_aabb = AABB();
    for (const SubMesh& sub_mesh : m_sub_meshes) {
        for (const shared::PackedTriangleVertex& vertex : sub_mesh.vertices) {
            m_local_aabb.expand(float3(vertex.position[0], vertex.position[1], vertex.position[2]));
        }
    }
}

void StaticMeshGeometry::update_render_state()
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
    std::vector<RenderHandle> geometries;
    for (const SubMesh& sub_mesh : m_sub_meshes) {
        RenderTriangleGeometryDesc desc = {};
        desc.vertex_count = sgl::narrow_cast<uint32_t>(sub_mesh.vertices.size());
        desc.index_count = sgl::narrow_cast<uint32_t>(sub_mesh.indices.size());
        RenderTriangleGeometryHandle geometry = render_scene->create_triangle_geometry(desc);
        RenderTriangleGeometryData data = {};
        data.vertices = sub_mesh.vertices;
        data.indices = sub_mesh.indices;
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

FALCOR_SCENE_REGISTER_GEOMETRY(StaticMeshGeometry);

} // namespace falcor
