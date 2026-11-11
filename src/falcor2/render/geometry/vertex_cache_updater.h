// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "geometry_updater.h"
#include "falcor2/render/shared_geometry_types.h"
#include "falcor2/utils/managed_vector.h"

#include <sgl/device/kernel.h>

namespace falcor {

template<typename Vertex>
class VertexCacheDeformer;

/// Prepares registered cache deformers and executes one shared batch per output format.
/// Owns work storage, but neither deformers nor their source/output geometry.
template<typename Vertex>
class VertexCacheUpdater : public GeometryUpdater {
public:
    using VertexType = Vertex;

    VertexCacheUpdater();
    ~VertexCacheUpdater() override;

    void register_deformer(VertexCacheDeformer<Vertex>& deformer);
    void unregister_deformer(VertexCacheDeformer<Vertex>& deformer);

    void prepare(sgl::CommandEncoder* encoder, double time, bool initialize) override;
    bool has_work() const override { return m_work_count != 0; }

    void append(const shared::GeometryAnimationWork& work)
    {
        m_work.ensure_size(m_work_count + 1);
        m_work.data()[m_work_count++] = work;
    }

    /// Upload and execute the collected work. Outputs must be in unordered_access state.
    void dispatch(sgl::CommandEncoder* encoder) override;

private:
    std::vector<VertexCacheDeformer<Vertex>*> m_deformers;
    ref<sgl::ComputeKernel> m_kernel;
    ManagedVector<shared::GeometryAnimationWork> m_work;
    size_t m_work_count = 0;
};

// ELF visibility must be declared before these types are instantiated, including their vtables and RTTI.
// MSVC applies dllexport at the definition instead; extern dllexport is not supported.
#if !SGL_WINDOWS || !defined(FALCOR_DLL)
extern template class FALCOR_API VertexCacheUpdater<shared::PackedTriangleVertex>;
extern template class FALCOR_API VertexCacheUpdater<shared::PackedLSSVertex>;
#endif

} // namespace falcor
