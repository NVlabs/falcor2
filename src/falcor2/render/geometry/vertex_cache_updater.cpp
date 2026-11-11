// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vertex_cache_updater.h"
#include "vertex_cache_deformer.h"
#include "falcor2/render/utils/geometry_dispatch.h"

#include <sgl/device/shader_cursor.h>
#include <sgl/core/type_utils.h>
#include <algorithm>
#include <type_traits>

namespace falcor {

template<typename Vertex>
VertexCacheUpdater<Vertex>::VertexCacheUpdater()
    : GeometryUpdater(GeometryUpdatePhase::vertex_cache)
    , m_work(
          {.label = std::is_same_v<Vertex, shared::PackedTriangleVertex> ? "TriangleGeometryData::animation_work"
                                                                         : "LSSGeometryData::animation_work"}
      )
{
}

template<typename Vertex>
VertexCacheUpdater<Vertex>::~VertexCacheUpdater()
{
    // Retained geometries can outlive the scene. Their destructors must not call a destroyed updater.
    for (auto* deformer : m_deformers) {
        deformer->m_updater = nullptr;
        deformer->m_prepare_output = {};
    }
}

template<typename Vertex>
void VertexCacheUpdater<Vertex>::register_deformer(VertexCacheDeformer<Vertex>& deformer)
{
    FALCOR_ASSERT(!deformer.m_updater);
    m_deformers.push_back(&deformer);
}

template<typename Vertex>
void VertexCacheUpdater<Vertex>::unregister_deformer(VertexCacheDeformer<Vertex>& deformer)
{
    const auto it = std::find(m_deformers.begin(), m_deformers.end(), &deformer);
    FALCOR_ASSERT(it != m_deformers.end());
    m_deformers.erase(it);
}

template<typename Vertex>
void VertexCacheUpdater<Vertex>::prepare(sgl::CommandEncoder* encoder, double time, bool initialize)
{
    m_work_count = 0;
    for (auto* deformer : m_deformers)
        deformer->prepare(encoder, time, initialize);
}

template<typename Vertex>
void VertexCacheUpdater<Vertex>::dispatch(sgl::CommandEncoder* encoder)
{
    m_work.resize(m_work_count);
    if (!has_work())
        return;
    m_work.mark_all_dirty();
    if (!m_kernel)
        m_kernel = encoder->device()->create_compute_kernel({
            .program = encoder->device()->load_program(
                "falcor2/render/kernels/geometry_animation.slang",
                {std::is_same_v<Vertex, shared::PackedTriangleVertex> ? "animate_mesh" : "animate_lss"}
            ),
        });
    m_work.update_buffer(encoder);
    uint32_t width = 0;
    for (const auto& item : std::span(m_work.data(), m_work.size()))
        width = std::max(width, item.vertex_count);
    const uint32_t height = sgl::narrow_cast<uint32_t>(m_work.size());
    const uint2 max_tile_size = geometry_dispatch_max_tile_size(m_kernel);
    for (uint32_t y = 0; y < height;) {
        const uint32_t count_y = std::min(height - y, max_tile_size.y);
        for (uint32_t x = 0; x < width;) {
            const uint32_t count_x = std::min(width - x, max_tile_size.x);
            m_kernel->dispatch(
                uint3(count_x, count_y, 1),
                [&](sgl::ShaderCursor cursor)
                {
                    cursor = cursor.find_entry_point(0);
                    cursor["work"] = m_work.buffer();
                    cursor["thread_offset"] = uint2(x, y);
                },
                encoder
            );
            x += count_x;
        }
        y += count_y;
    }
}

#if SGL_WINDOWS
template class FALCOR_API VertexCacheUpdater<shared::PackedTriangleVertex>;
template class FALCOR_API VertexCacheUpdater<shared::PackedLSSVertex>;
#else
template class VertexCacheUpdater<shared::PackedTriangleVertex>;
template class VertexCacheUpdater<shared::PackedLSSVertex>;
#endif

static_assert(sizeof(shared::GeometryAnimationWork) == 48);
} // namespace falcor
