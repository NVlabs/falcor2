// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/animation_timing.h"
#include "falcor2/render/shared_geometry_types.h"
#include "falcor2/utils/aabb.h"
#include "falcor2/utils/buffer_ptr.h"

#include <sgl/core/short_vector.h>

#include <functional>

namespace falcor {

class RenderScene;

/// Final output range; pointers address its first vertex. Order matches geometry setup.
struct GeometryDeformationTarget {
    RWBufferPtr current;
    RWBufferPtr previous;
    uint32_t vertex_count;
};

/// Produces fixed-topology geometry in the Vertex output format. Owns or references all inputs;
/// their formats and counts need not match the output. Exclusively owned by one dynamic geometry.
/// Configuration, output layout and animation range stay fixed until the geometry is reconfigured.
template<typename Vertex>
class GeometryDeformer {
public:
    /// Resolve current output ranges and notify geometry of the requested history update.
    /// The callback is retained until detach; it must not own the geometry.
    using PrepareOutput = std::function<
        void(shared::GeometryHistoryUpdate history_update, sgl::short_vector<GeometryDeformationTarget, 1>& targets)>;

    virtual ~GeometryDeformer() = default;

    /// Register after render outputs exist. Replacement/removal must detach before destroying outputs.
    virtual void attach(RenderScene& render_scene, PrepareOutput prepare_output) = 0;
    virtual void detach() = 0;

    /// Select inputs, decide history updates and append work for later batched execution.
    /// Check inputs even at unchanged time; initialize forces both current and previous outputs.
    /// Resolve output ranges through PrepareOutput only when work is needed; do not retain them.
    /// Declare source resource states through encoder to retain inputs through GPU execution.
    /// The render collection transitions final output buffers and dispatches the batch afterwards.
    virtual void prepare(sgl::CommandEncoder* encoder, double time, bool initialize) = 0;

    /// Conservative bound for the final result in geometry-local space, available after setup and during PrepareOutput.
    virtual AABB local_aabb() const = 0;
    virtual AnimationTimeRange animation_range() const = 0;
    virtual bool has_animation() const = 0;
};

} // namespace falcor
