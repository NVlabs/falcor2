// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"
#include "falcor2/core/object.h"
#include "falcor2/render/shared_emissive_geometry_types.h"
#include "falcor2/render/shared_emissive_triangle_tree_types.h"
#include "falcor2/utils/aabb.h"

#include <sgl/device/fwd.h>

#include <cstdint>
#include <span>
#include <vector>

namespace falcor {

class EmissiveGeometrySystem;

enum class EmissiveTriangleTreeSplitHeuristic : uint32_t {
    equal,
    binned_sah,
    binned_saoh,
};
SGL_ENUM_INFO(
    EmissiveTriangleTreeSplitHeuristic,
    {
        {EmissiveTriangleTreeSplitHeuristic::equal, "equal"},
        {EmissiveTriangleTreeSplitHeuristic::binned_sah, "binned_sah"},
        {EmissiveTriangleTreeSplitHeuristic::binned_saoh, "binned_saoh"},
    }
);
SGL_ENUM_REGISTER(EmissiveTriangleTreeSplitHeuristic);

/// CPU-built binary hierarchy over active emissive triangles.
class FALCOR_API EmissiveTriangleTree {
public:
    struct BuildOptions {
        EmissiveTriangleTreeSplitHeuristic split_heuristic{EmissiveTriangleTreeSplitHeuristic::binned_sah};
        uint32_t max_triangle_count_per_leaf{8};
        uint32_t bin_count{16};
    };

    struct Stats {
        uint32_t node_count{0};
        uint32_t leaf_count{0};
        uint32_t max_depth{0};
    };

    explicit EmissiveTriangleTree(sgl::Device* device);

    void build(const EmissiveGeometrySystem& emissive_geometry, const BuildOptions& options);

    /// Recomputes node attributes while preserving the topology created by build().
    /// The active emissive triangle count, IDs, and ordering must match the preceding successful build.
    /// Geometry and flux may change; callers must rebuild when the active triangle topology changes.
    void refit(const EmissiveGeometrySystem& emissive_geometry, sgl::CommandEncoder* command_encoder = nullptr);
    void write_to_cursor(sgl::ShaderCursor cursor) const;

    const Stats& stats() const { return m_stats; }
    uint32_t refit_dispatch_count() const { return m_refit_dispatch_count; }
    uint64_t node_buffer_recreation_count() const { return m_node_buffer_recreation_count; }

private:
    struct BuildTriangle {
        shared::EmissiveTriangleID id;
        uint32_t active_index;
        float3 bounds_min;
        float3 bounds_max;
        float3 center;
        float3 normal;
        float flux;
    };

    struct RefitRange {
        uint32_t offset{0};
        uint32_t count{0};
    };

    struct BuildAggregate {
        AABB bounds;
        uint32_t triangle_count{0};
        float flux{0.f};
        float3 weighted_normal_sum{0.f};
        float3 cone_direction{0.f};
        float cos_cone_angle{-1.f};
    };

    uint32_t build_node(
        std::vector<BuildTriangle>& triangles,
        uint32_t begin,
        uint32_t end,
        uint32_t depth,
        const BuildOptions& options
    );
    uint32_t
    partition(std::vector<BuildTriangle>& triangles, uint32_t begin, uint32_t end, const BuildOptions& options) const;
    BuildAggregate compute_aggregate(std::span<const BuildTriangle> triangles) const;
    BuildAggregate merge_aggregates(const BuildAggregate& left, const BuildAggregate& right) const;
    shared::EmissiveTriangleTreeNode node_data_from_aggregate(const BuildAggregate& aggregate) const;
    BuildAggregate decoded_aggregate(uint32_t node_index) const;
    void create_refit_kernels();
    void build_refit_schedule();
    void upload_nodes();
    void upload();

    ref<sgl::Device> m_device;
    std::vector<shared::EmissiveTriangleTreeNode> m_nodes;
    std::vector<BuildAggregate> m_build_aggregates;
    std::vector<std::vector<uint32_t>> m_internal_node_indices_by_depth;
    std::vector<uint32_t> m_leaf_node_indices;
    std::vector<uint32_t> m_refit_node_indices;
    std::vector<RefitRange> m_internal_refit_ranges;
    RefitRange m_leaf_refit_range;
    std::vector<shared::EmissiveTriangleID> m_triangle_ids;
    std::vector<uint32_t> m_active_triangle_leaf_indices;
    Stats m_stats;
    uint32_t m_refit_dispatch_count{0};
    uint64_t m_node_buffer_recreation_count{0};
    bool m_built{false};

    ref<sgl::ComputeKernel> m_refit_leaves_kernel;
    ref<sgl::ComputeKernel> m_refit_internal_nodes_kernel;
    ref<sgl::Buffer> m_nodes_buffer;
    ref<sgl::Buffer> m_refit_weighted_normal_sums_buffer;
    ref<sgl::Buffer> m_refit_node_indices_buffer;
    ref<sgl::Buffer> m_triangle_ids_buffer;
    ref<sgl::Buffer> m_active_triangle_leaf_indices_buffer;
};

} // namespace falcor
