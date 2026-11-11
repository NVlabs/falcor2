// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/geometry.h"
#include "geometry_vertex_attribute.h"
#include "geometry_deformer.h"

#include <memory>

namespace falcor {

/// Fixed-topology geometry whose final vertices are produced by one deformer on the GPU.
class FALCOR_API DynamicCurveGeometry : public Geometry {
    FALCOR_SCENE_OBJECT(DynamicCurveGeometry, Geometry)
public:
    /// Replace prepared topology/static attributes and its deformer together; resets vertex history.
    /// The geometry must belong to a Scene. The deformer must produce these fixed output ranges.
    void set_curve_data(
        std::vector<shared::PackedLSSVertex> vertices,
        std::vector<uint32_t> indices,
        std::vector<GeometryVertexAttribute> vertex_attributes,
        std::unique_ptr<GeometryDeformer<shared::PackedLSSVertex>> deformer
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

    std::vector<shared::PackedLSSVertex> m_vertices;
    std::vector<uint32_t> m_indices;
    std::vector<GeometryVertexAttribute> m_vertex_attributes;

    RenderLSSGeometryHandle m_render_geometry;

    std::unique_ptr<GeometryDeformer<shared::PackedLSSVertex>> m_deformer;
};

} // namespace falcor
