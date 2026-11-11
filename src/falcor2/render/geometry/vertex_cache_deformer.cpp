// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vertex_cache_deformer.h"
#include "curve_geometry_data.h"
#include "falcor2/render/render_scene.h"

#include <cstring>

namespace falcor {

template<typename Vertex>
void VertexCacheDeformer<Vertex>::attach(
    RenderScene& render_scene,
    typename GeometryDeformer<Vertex>::PrepareOutput prepare_output
)
{
    FALCOR_ASSERT(!m_updater);
    auto& updater = render_scene.get_or_create_geometry_updater<VertexCacheUpdater<Vertex>>();
    m_prepare_output = std::move(prepare_output);
    m_history = {};
    updater.register_deformer(*this);
    m_updater = &updater;
}

#if SGL_WINDOWS
template class FALCOR_API VertexCacheDeformer<shared::PackedTriangleVertex>;
template class FALCOR_API VertexCacheDeformer<shared::PackedLSSVertex>;
#else
template class VertexCacheDeformer<shared::PackedTriangleVertex>;
template class VertexCacheDeformer<shared::PackedLSSVertex>;
#endif

std::unique_ptr<VertexCacheDeformer<shared::PackedTriangleVertex>> create_vertex_cache(
    sgl::Device* device,
    std::span<const StaticMeshGeometryDataDesc> samples,
    std::span<const float> times,
    AnimationInterpolationMode interpolation,
    std::vector<MeshGeometryData>& sub_meshes,
    std::span<const uint32_t> sample_indices
)
{
    validate_geometry_samples(times, samples.size(), interpolation, sample_indices);
    const auto& desc = samples.front();
    for (const auto& sample : samples)
        FALCOR_CHECK(sample.vertex_count == desc.vertex_count, "Motion sample vertex count differs");
    AABB local_aabb;
    sgl::short_vector<GeometrySampleStorage<shared::PackedTriangleVertex>, 1> sources;
    std::vector<std::vector<uint32_t>> remaps;
    make_mesh_geometry_data(desc, sub_meshes, &remaps);
    for (size_t submesh = 0; submesh < sub_meshes.size(); ++submesh) {
        const auto& remap = remaps[submesh];
        std::vector<shared::PackedTriangleVertex> vertices(samples.size() * remap.size());
        for (size_t sample = 0; sample < samples.size(); ++sample) {
            auto output = std::span(vertices).subspan(sample * remap.size(), remap.size());
            pack_mesh_vertices(samples[sample].vertex_attributes, remap, output);
            for (const auto& vertex : output)
                local_aabb.expand(vertex.position);
        }
        sources.emplace_back().reset(device, std::move(vertices), sgl::narrow_cast<uint32_t>(remap.size()));
    }
    return std::make_unique<VertexCacheDeformer<shared::PackedTriangleVertex>>(
        std::move(sources),
        times,
        interpolation,
        sample_indices,
        local_aabb
    );
}

std::unique_ptr<VertexCacheDeformer<shared::PackedLSSVertex>> create_vertex_cache(
    sgl::Device* device,
    std::span<const StaticCurveGeometryDataDesc> samples,
    std::span<const float> times,
    AnimationInterpolationMode interpolation,
    std::vector<shared::PackedLSSVertex>& base_vertices,
    std::vector<uint32_t>& indices,
    std::vector<GeometryVertexAttribute>& vertex_attributes,
    std::span<const uint32_t> sample_indices
)
{
    validate_geometry_samples(times, samples.size(), interpolation, sample_indices);
    const auto& desc = samples.front();
    for (const auto& sample : samples)
        FALCOR_CHECK(sample.vertex_count == desc.vertex_count, "Motion sample vertex count differs");
    AABB local_aabb;
    sgl::short_vector<GeometrySampleStorage<shared::PackedLSSVertex>, 1> sources;
    std::vector<uint32_t> remap;
    make_curve_geometry_data(desc, base_vertices, indices, vertex_attributes, &remap);
    std::vector<shared::PackedLSSVertex> vertices(samples.size() * remap.size());
    for (size_t sample = 0; sample < samples.size(); ++sample) {
        for (size_t i = 0; i < remap.size(); ++i) {
            auto vertex = base_vertices[i];
            const auto& positions = samples[sample].position_stream;
            const auto& radii = samples[sample].radius_stream;
            vertex.position = float3(0.f);
            vertex.radius = 0.f;
            if (positions.data)
                std::memcpy(
                    &vertex.position,
                    reinterpret_cast<const uint8_t*>(positions.data) + remap[i] * positions.stride,
                    sizeof(float3)
                );
            if (radii.data)
                std::memcpy(
                    &vertex.radius,
                    reinterpret_cast<const uint8_t*>(radii.data) + remap[i] * radii.stride,
                    sizeof(float)
                );
            vertices[sample * remap.size() + i] = vertex;
            local_aabb.expand(vertex.position, vertex.radius);
        }
    }
    sources.emplace_back().reset(device, std::move(vertices), sgl::narrow_cast<uint32_t>(remap.size()));
    return std::make_unique<VertexCacheDeformer<shared::PackedLSSVertex>>(
        std::move(sources),
        times,
        interpolation,
        sample_indices,
        local_aabb
    );
}

} // namespace falcor
