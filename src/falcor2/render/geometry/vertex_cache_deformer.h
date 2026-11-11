// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "geometry_deformer.h"
#include "geometry_animation.h"
#include "vertex_cache_updater.h"
#include "geometry_sample_storage.h"
#include "static_mesh_geometry.h"
#include "static_curve_geometry.h"

#include <sgl/core/short_vector.h>

namespace falcor {

/// Interpolates stored packed vertices into final output ranges. Each store has the same sample count
/// and the vertex order/count of its corresponding target. Setup helpers validate and pack those samples.
template<typename Vertex>
class VertexCacheDeformer : public GeometryDeformer<Vertex> {
public:
    VertexCacheDeformer(
        sgl::short_vector<GeometrySampleStorage<Vertex>, 1> sources,
        std::span<const float> times,
        AnimationInterpolationMode interpolation,
        std::span<const uint32_t> sample_indices,
        const AABB& local_aabb
    )
        : m_sources(std::move(sources))
        , m_times(times.begin(), times.end())
        , m_sample_indices(sample_indices.begin(), sample_indices.end())
        , m_interpolation(interpolation)
        , m_local_aabb(local_aabb)
    {
        FALCOR_ASSERT(!m_times.empty());
    }

    ~VertexCacheDeformer() override { detach(); }

    void attach(RenderScene& render_scene, typename GeometryDeformer<Vertex>::PrepareOutput prepare_output) override;

    void detach() override
    {
        if (m_updater) {
            m_updater->unregister_deformer(*this);
            m_updater = nullptr;
        }
        m_prepare_output = {};
    }

    void prepare(sgl::CommandEncoder* encoder, double time, bool initialize) override
    {
        FALCOR_ASSERT(m_updater);
        const auto selection = select_geometry_samples(time, m_times, m_interpolation, m_sample_indices);
        const bool changed = selection != m_selection;
        m_selection = selection;
        if (initialize)
            m_history = {};
        const auto history_update = m_history.update(changed);
        if (history_update == shared::GeometryHistoryUpdate::none)
            return;
        sgl::short_vector<GeometryDeformationTarget, 1> targets;
        m_prepare_output(history_update, targets);
        FALCOR_ASSERT(targets.size() == m_sources.size());
        for (size_t i = 0; i < targets.size(); ++i) {
            const auto sources = m_sources[i].prepare(encoder, m_selection.source0, m_selection.source1);
            m_updater->append({
                .source0 = to_buffer_ptr(sources[0].buffer, sizeof(Vertex) * sources[0].element_offset),
                .source1 = to_buffer_ptr(sources[1].buffer, sizeof(Vertex) * sources[1].element_offset),
                .current = targets[i].current,
                .previous = targets[i].previous,
                .vertex_count = targets[i].vertex_count,
                .source1_weight = m_selection.source1_weight,
                .history_update = history_update,
            });
        }
    }

    AABB local_aabb() const override { return m_local_aabb; }
    AnimationTimeRange animation_range() const override { return {m_times.front(), m_times.back()}; }
    bool has_animation() const override { return m_times.size() > 1; }

private:
    friend class VertexCacheUpdater<Vertex>;
    VertexCacheUpdater<Vertex>* m_updater = nullptr;
    sgl::short_vector<GeometrySampleStorage<Vertex>, 1> m_sources;
    std::vector<float> m_times;
    std::vector<uint32_t> m_sample_indices;
    AnimationInterpolationMode m_interpolation;
    GeometryAnimationSelection m_selection;
    GeometryAnimationHistory m_history;
    typename GeometryDeformer<Vertex>::PrepareOutput m_prepare_output;
    AABB m_local_aabb;

    FALCOR_NON_COPYABLE_AND_MOVABLE(VertexCacheDeformer);
};

// ELF visibility must be declared before these types are instantiated, including their vtables and RTTI.
// MSVC applies dllexport at the definition instead; extern dllexport is not supported.
#if !SGL_WINDOWS || !defined(FALCOR_DLL)
extern template class FALCOR_API VertexCacheDeformer<shared::PackedTriangleVertex>;
extern template class FALCOR_API VertexCacheDeformer<shared::PackedLSSVertex>;
#endif

/// Pack samples and return an unattached vertex-cache deformer; replace sub_meshes with its matching topology.
/// Topology/static attributes come from the first sample; all samples share its vertex order and count.
/// @param times Finite nondecreasing times in seconds; paired equal times encode a jump (left, right).
/// @param interpolation Step or linear interpolation, clamped outside the time range.
/// @param sub_meshes Output topology and initial vertices to install with the returned deformer.
/// @param sample_indices Stored sample index per time; empty means one sample per time.
FALCOR_API std::unique_ptr<VertexCacheDeformer<shared::PackedTriangleVertex>> create_vertex_cache(
    sgl::Device* device,
    std::span<const StaticMeshGeometryDataDesc> samples,
    std::span<const float> times,
    AnimationInterpolationMode interpolation,
    std::vector<MeshGeometryData>& sub_meshes,
    std::span<const uint32_t> sample_indices = {}
);

/// Curve counterpart, including radii in the bounds. Replaces vertices, indices and vertex_attributes
/// with the matching output data to install with the returned deformer.
FALCOR_API std::unique_ptr<VertexCacheDeformer<shared::PackedLSSVertex>> create_vertex_cache(
    sgl::Device* device,
    std::span<const StaticCurveGeometryDataDesc> samples,
    std::span<const float> times,
    AnimationInterpolationMode interpolation,
    std::vector<shared::PackedLSSVertex>& vertices,
    std::vector<uint32_t>& indices,
    std::vector<GeometryVertexAttribute>& vertex_attributes,
    std::span<const uint32_t> sample_indices = {}
);

} // namespace falcor
