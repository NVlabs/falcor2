// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/geometry.h"
#include "mesh_geometry_data.h"
#include "falcor2/render/geometry/geometry_vertex_attribute.h"

#include "falcor2/core/types.h"
#include "falcor2/core/properties.h"

#include <span>
#include <vector>

namespace falcor {

struct StaticMeshGeometryDataDesc {
    template<typename T>
    struct IndexStream {
        const T* data;
        size_t stride;
    };

    struct SubMesh {
        std::string_view name;
        uint32_t index_count;
        IndexStream<uint32_t> index_stream;
    };

    std::string_view name;
    uint32_t vertex_count;
    std::span<const MeshVertexAttributeDesc> vertex_attributes;

    std::vector<SubMesh> sub_meshes;
};

/// Static mesh geometry.
/// A static mesh geometry contains one or more sub-meshes with triangle data.
/// Each sub-mesh has its own material (which can be overriden per geometry instance).
class FALCOR_API StaticMeshGeometry : public Geometry {
    FALCOR_SCENE_OBJECT(StaticMeshGeometry, Geometry)
public:
    ~StaticMeshGeometry() override;

    /// Set the mesh data.
    /// @param desc Description of the mesh data.
    void set_mesh_data(const StaticMeshGeometryDataDesc& desc);

    // SceneObject interface

    void on_add_to_scene() override;
    void on_remove_from_scene() override { clear_render_state(); }

    // Geometry interface

    virtual void update_render_state() override;

    /// Number of sub-meshes in this geometry.
    size_t sub_mesh_count() const { return m_sub_meshes.size(); }

    /// Packed vertices of the given sub-mesh (CPU-side).
    std::span<const shared::PackedTriangleVertex> vertices(size_t sub_mesh_index) const
    {
        return m_sub_meshes.at(sub_mesh_index).vertices;
    }

    /// Packed vertices of the given sub-mesh (CPU-side, mutable).
    std::span<shared::PackedTriangleVertex> mutable_vertices(size_t sub_mesh_index)
    {
        return m_sub_meshes.at(sub_mesh_index).vertices;
    }

    /// Number of vertices in the given sub-mesh.
    size_t vertex_count(size_t sub_mesh_index) const { return vertices(sub_mesh_index).size(); }

    /// Triangle indices of the given sub-mesh (CPU-side).
    std::span<const uint32_t> indices(size_t sub_mesh_index) const { return m_sub_meshes.at(sub_mesh_index).indices; }

    /// Triangle indices of the given sub-mesh (CPU-side, mutable).
    std::span<uint32_t> mutable_indices(size_t sub_mesh_index) { return m_sub_meshes.at(sub_mesh_index).indices; }

    /// Number of indices in the given sub-mesh.
    size_t index_count(size_t sub_mesh_index) const { return indices(sub_mesh_index).size(); }

    /// Vertex attributes of the given sub-mesh in its remapped vertex order.
    std::span<const MeshVertexAttribute> vertex_attributes(size_t sub_mesh_index) const
    {
        return m_sub_meshes.at(sub_mesh_index).vertex_attributes;
    }

    /// Find a vertex attribute of the given sub-mesh by semantic and index.
    MeshVertexAttribute* find_vertex_attribute(size_t sub_mesh_index, MeshAttributeSemantic semantic, uint32_t index);

    /// Find a vertex attribute of the given sub-mesh by semantic and index.
    const MeshVertexAttribute*
    find_vertex_attribute(size_t sub_mesh_index, MeshAttributeSemantic semantic, uint32_t index) const;

    /// Recompute local bounds from the CPU-side packed vertex positions.
    void recompute_local_aabb();

    /// Flag vertex data dirty so update_render_state() re-uploads it on the next scene update.
    void mark_vertices_dirty() { mark_dirty(DirtyFlags::render_state); }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        FALCOR_UNUSED(r);
    }

private:
    void clear_render_state();
    void register_geometry_attributes();

    using SubMesh = MeshGeometryData;

    std::vector<SubMesh> m_sub_meshes;
    std::vector<RenderTriangleGeometryHandle> m_render_geometries;

    bool m_dirty_data{false};
};

} // namespace falcor
