// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/geometry.h"

#include "falcor2/core/types.h"
#include "falcor2/core/properties.h"

#include <span>

namespace falcor {

struct StaticMeshGeometryDataDesc {
    template<typename T>
    struct VertexAttributeStream {
        const T* data;
        size_t stride;
    };

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
    VertexAttributeStream<float3> position_stream;
    VertexAttributeStream<float3> normal_stream;
    VertexAttributeStream<float3> tangent_stream;
    VertexAttributeStream<float> handedness_stream;
    VertexAttributeStream<float2> texcoord_stream;

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

    virtual void clear_invalid_references() override;

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
    struct SubMesh {
        std::string name;
        std::vector<shared::PackedTriangleVertex> vertices;
        std::vector<uint32_t> indices;
        Material* material{nullptr};
    };

    std::vector<SubMesh> m_sub_meshes;
    std::vector<RenderTriangleGeometryHandle> m_render_geometries;

    bool m_dirty_data{false};
};

} // namespace falcor
