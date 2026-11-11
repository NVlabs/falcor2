// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/geometry.h"
#include "mesh_geometry_data.h"
#include "geometry_deformer.h"

#include <memory>

namespace falcor {

/// Fixed-topology geometry whose final vertices are produced by one deformer on the GPU.
class FALCOR_API DynamicMeshGeometry : public Geometry {
    FALCOR_SCENE_OBJECT(DynamicMeshGeometry, Geometry)
public:
    /// Replace prepared topology/static attributes and its deformer together; resets vertex history.
    /// The geometry must belong to a Scene. The deformer must produce these fixed output ranges.
    void set_mesh_data(
        std::vector<MeshGeometryData> sub_meshes,
        std::unique_ptr<GeometryDeformer<shared::PackedTriangleVertex>> deformer
    );

    // SceneObject interface

    void on_add_to_scene() override;
    void on_remove_from_scene() override;

    // Geometry interface

    void update_render_state() override;

    AnimationTimeRange animation_range() const
    {
        return m_deformer ? m_deformer->animation_range() : AnimationTimeRange{};
    }
    bool has_animation() const { return m_deformer && m_deformer->has_animation(); }

    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        FALCOR_UNUSED(r);
    }

private:
    void prepare_output(
        shared::GeometryHistoryUpdate history_update,
        sgl::short_vector<GeometryDeformationTarget, 1>& targets
    );
    void clear_render_state();
    void register_geometry_attributes();

    using SubMesh = MeshGeometryData;

    std::vector<SubMesh> m_sub_meshes;
    std::vector<RenderTriangleGeometryHandle> m_render_geometries;

    std::unique_ptr<GeometryDeformer<shared::PackedTriangleVertex>> m_deformer;
};

} // namespace falcor
