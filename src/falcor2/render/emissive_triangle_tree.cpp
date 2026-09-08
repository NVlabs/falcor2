// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "emissive_triangle_tree.h"

#include "falcor2/core/error.h"
#include "falcor2/render/emissive_geometry_system.h"
#include "falcor2/utils/aabb.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>
#include <sgl/device/kernel.h>
#include <sgl/device/shader.h>
#include <sgl/device/shader_cursor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace falcor {
namespace {

constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();
constexpr float INVALID_COS_CONE_ANGLE = -1.f;

float surface_area(const AABB& bounds)
{
    if (!bounds.is_valid())
        return 0.f;
    const float3 size = bounds.size();
    const float area = 2.f * (size.x * size.y + size.y * size.z + size.z * size.x);
    return std::isfinite(area) && area >= 0.f ? area : std::numeric_limits<float>::infinity();
}

uint32_t largest_axis(const float3& extent)
{
    if (extent.x >= extent.y && extent.x >= extent.z)
        return 0;
    return extent.y >= extent.z ? 1 : 2;
}

float component(const float3& value, uint32_t axis)
{
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

float safe_acos(float value)
{
    return std::acos(std::clamp(value, -1.f, 1.f));
}

float orientation_cost(float cos_cone_angle)
{
    constexpr float pi = std::numbers::pi_v<float>;
    constexpr float half_pi = 0.5f * pi;
    constexpr float two_pi = 2.f * pi;

    const float theta_o = cos_cone_angle == INVALID_COS_CONE_ANGLE ? pi : safe_acos(cos_cone_angle);
    const float theta_w = std::min(theta_o + half_pi, pi);
    const float sin_theta_o = std::sin(theta_o);
    const float cos_theta_o = std::cos(theta_o);
    const float cost = two_pi * (1.f - cos_theta_o)
        + half_pi
            * (2.f * theta_w * sin_theta_o - std::cos(theta_o - 2.f * theta_w) - 2.f * theta_o * sin_theta_o
               + cos_theta_o);
    return std::isfinite(cost) && cost >= 0.f ? cost : std::numeric_limits<float>::infinity();
}

} // namespace

EmissiveTriangleTree::EmissiveTriangleTree(sgl::Device* device)
    : m_device(device)
{
    FALCOR_CHECK_NOT_NULL(m_device);
}

EmissiveTriangleTree::BuildAggregate
EmissiveTriangleTree::compute_aggregate(std::span<const BuildTriangle> triangles) const
{
    BuildAggregate aggregate;
    for (const BuildTriangle& triangle : triangles) {
        aggregate.bounds.expand(triangle.bounds_min);
        aggregate.bounds.expand(triangle.bounds_max);
        aggregate.flux += triangle.flux;
        aggregate.weighted_normal_sum += triangle.normal * std::max(triangle.flux, 1e-20f);
        ++aggregate.triangle_count;
    }

    const float cone_length = sgl::math::length(aggregate.weighted_normal_sum);
    if (std::isfinite(cone_length) && cone_length > 1e-20f) {
        aggregate.cone_direction = aggregate.weighted_normal_sum / cone_length;
        aggregate.cos_cone_angle = 1.f;
        for (const BuildTriangle& triangle : triangles) {
            aggregate.cos_cone_angle = std::min(
                aggregate.cos_cone_angle,
                std::clamp(sgl::math::dot(aggregate.cone_direction, triangle.normal), -1.f, 1.f)
            );
        }
    }
    return aggregate;
}

EmissiveTriangleTree::BuildAggregate
EmissiveTriangleTree::merge_aggregates(const BuildAggregate& left, const BuildAggregate& right) const
{
    if (left.triangle_count == 0)
        return right;
    if (right.triangle_count == 0)
        return left;

    BuildAggregate aggregate;
    aggregate.bounds = left.bounds;
    aggregate.bounds.expand(right.bounds);
    aggregate.triangle_count = left.triangle_count + right.triangle_count;
    aggregate.flux = left.flux + right.flux;
    aggregate.weighted_normal_sum = left.weighted_normal_sum + right.weighted_normal_sum;

    const float cone_length = sgl::math::length(aggregate.weighted_normal_sum);
    if (!std::isfinite(cone_length) || cone_length <= 1e-20f || left.cos_cone_angle == INVALID_COS_CONE_ANGLE
        || right.cos_cone_angle == INVALID_COS_CONE_ANGLE) {
        return aggregate;
    }

    aggregate.cone_direction = aggregate.weighted_normal_sum / cone_length;
    aggregate.cos_cone_angle = 1.f;
    for (const BuildAggregate* child : {&left, &right}) {
        const float child_cos_cone_angle = shared::detail::widen_emissive_triangle_tree_cone_cos(
            child->cos_cone_angle,
            sgl::math::dot(aggregate.cone_direction, child->cone_direction)
        );
        if (child_cos_cone_angle == INVALID_COS_CONE_ANGLE) {
            aggregate.cos_cone_angle = INVALID_COS_CONE_ANGLE;
            aggregate.cone_direction = float3(0.f);
            return aggregate;
        }
        aggregate.cos_cone_angle = std::min(aggregate.cos_cone_angle, child_cos_cone_angle);
    }
    return aggregate;
}

shared::EmissiveTriangleTreeNode EmissiveTriangleTree::node_data_from_aggregate(const BuildAggregate& aggregate) const
{
    shared::EmissiveTriangleTreeNode node{};
    node.set_bounds(aggregate.bounds.min, aggregate.bounds.max);
    node.flux = aggregate.flux;
    node.right_child = INVALID_INDEX;
    node.cone_direction = aggregate.cone_direction;
    node.cos_cone_angle = aggregate.cos_cone_angle;
    node.triangle_offset = 0;
    node.triangle_count = 0;
    return node;
}

EmissiveTriangleTree::BuildAggregate EmissiveTriangleTree::decoded_aggregate(uint32_t node_index) const
{
    const shared::PackedEmissiveTriangleTreeNode packed
        = shared::detail::pack_emissive_triangle_tree_node(m_nodes[node_index]);
    const shared::EmissiveTriangleTreeNode data = shared::detail::unpack_emissive_triangle_tree_node(packed);

    BuildAggregate aggregate;
    aggregate.bounds = AABB(data.get_bounds_min(), data.get_bounds_max());
    aggregate.triangle_count = m_build_aggregates[node_index].triangle_count;
    aggregate.flux = data.flux;
    aggregate.weighted_normal_sum = m_build_aggregates[node_index].weighted_normal_sum;
    aggregate.cone_direction = data.cone_direction;
    aggregate.cos_cone_angle = data.cos_cone_angle;
    return aggregate;
}

uint32_t EmissiveTriangleTree::partition(
    std::vector<BuildTriangle>& triangles,
    uint32_t begin,
    uint32_t end,
    const BuildOptions& options
) const
{
    AABB center_bounds;
    for (uint32_t i = begin; i < end; ++i)
        center_bounds.expand(triangles[i].center);

    const uint32_t fallback_axis = largest_axis(center_bounds.size());
    const uint32_t middle = begin + (end - begin) / 2;
    auto equal_partition = [&](uint32_t axis)
    {
        std::nth_element(
            triangles.begin() + begin,
            triangles.begin() + middle,
            triangles.begin() + end,
            [axis](const BuildTriangle& a, const BuildTriangle& b)
            {
                return component(a.center, axis) < component(b.center, axis);
            }
        );
        return middle;
    };

    if (options.split_heuristic == EmissiveTriangleTreeSplitHeuristic::equal)
        return equal_partition(fallback_axis);

    const uint32_t bin_count = std::clamp(options.bin_count, 2u, 128u);
    float best_cost = std::numeric_limits<float>::infinity();
    uint32_t best_axis = INVALID_INDEX;
    uint32_t best_bin = 0;

    auto split_cost = [&](const BuildAggregate& aggregate)
    {
        const float area = surface_area(aggregate.bounds);
        const float cost = aggregate.flux * area * orientation_cost(aggregate.cos_cone_angle);
        return std::isfinite(cost) && cost >= 0.f ? cost : std::numeric_limits<float>::infinity();
    };

    for (uint32_t axis = 0; axis < 3; ++axis) {
        const float min_center = component(center_bounds.min, axis);
        const float extent = component(center_bounds.size(), axis);
        if (!std::isfinite(min_center) || !std::isfinite(extent) || extent <= 1e-20f)
            continue;

        bool valid_axis = true;
        auto bin_index_for_triangle = [&](const BuildTriangle& triangle)
        {
            const float normalized = (component(triangle.center, axis) - min_center) / extent;
            if (!std::isfinite(normalized))
                return INVALID_INDEX;
            const float clamped_normalized = std::clamp(normalized, 0.f, 1.f);
            return std::min(static_cast<uint32_t>(clamped_normalized * float(bin_count)), bin_count - 1);
        };

        if (options.split_heuristic == EmissiveTriangleTreeSplitHeuristic::binned_sah) {
            struct BoundsAggregate {
                AABB bounds;
                uint32_t triangle_count{0};
            };
            auto merge_bounds = [](const BoundsAggregate& left, const BoundsAggregate& right)
            {
                if (left.triangle_count == 0)
                    return right;
                if (right.triangle_count == 0)
                    return left;
                BoundsAggregate result = left;
                result.bounds.expand(right.bounds);
                result.triangle_count += right.triangle_count;
                return result;
            };

            std::vector<BoundsAggregate> bins(bin_count);
            for (uint32_t i = begin; i < end; ++i) {
                const uint32_t bin_index = bin_index_for_triangle(triangles[i]);
                if (bin_index == INVALID_INDEX) {
                    valid_axis = false;
                    break;
                }
                bins[bin_index].bounds.expand(triangles[i].bounds_min);
                bins[bin_index].bounds.expand(triangles[i].bounds_max);
                ++bins[bin_index].triangle_count;
            }
            if (!valid_axis)
                continue;

            std::vector<BoundsAggregate> left_aggregates(bin_count - 1);
            std::vector<BoundsAggregate> right_aggregates(bin_count - 1);
            BoundsAggregate aggregate;
            for (uint32_t i = 0; i + 1 < bin_count; ++i) {
                aggregate = merge_bounds(aggregate, bins[i]);
                left_aggregates[i] = aggregate;
            }
            aggregate = {};
            for (uint32_t i = bin_count - 1; i > 0; --i) {
                aggregate = merge_bounds(aggregate, bins[i]);
                right_aggregates[i - 1] = aggregate;
            }
            for (uint32_t i = 0; i + 1 < bin_count; ++i) {
                if (left_aggregates[i].triangle_count == 0 || right_aggregates[i].triangle_count == 0)
                    continue;
                const float cost = surface_area(left_aggregates[i].bounds) * float(left_aggregates[i].triangle_count)
                    + surface_area(right_aggregates[i].bounds) * float(right_aggregates[i].triangle_count);
                if (std::isfinite(cost) && cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_bin = i;
                }
            }
            continue;
        }

        std::vector<BuildAggregate> bins(bin_count);
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t bin_index = bin_index_for_triangle(triangles[i]);
            if (bin_index == INVALID_INDEX) {
                valid_axis = false;
                break;
            }
            const BuildAggregate triangle_aggregate
                = compute_aggregate(std::span<const BuildTriangle>(&triangles[i], 1));
            bins[bin_index] = merge_aggregates(bins[bin_index], triangle_aggregate);
        }
        if (!valid_axis)
            continue;

        std::vector<BuildAggregate> left_aggregates(bin_count - 1);
        std::vector<BuildAggregate> right_aggregates(bin_count - 1);

        BuildAggregate aggregate;
        for (uint32_t i = 0; i + 1 < bin_count; ++i) {
            aggregate = merge_aggregates(aggregate, bins[i]);
            left_aggregates[i] = aggregate;
        }

        aggregate = {};
        for (uint32_t i = bin_count - 1; i > 0; --i) {
            aggregate = merge_aggregates(aggregate, bins[i]);
            right_aggregates[i - 1] = aggregate;
        }

        for (uint32_t i = 0; i + 1 < bin_count; ++i) {
            if (left_aggregates[i].triangle_count == 0 || right_aggregates[i].triangle_count == 0)
                continue;
            const float cost = split_cost(left_aggregates[i]) + split_cost(right_aggregates[i]);
            if (std::isfinite(cost) && cost < best_cost) {
                best_cost = cost;
                best_axis = axis;
                best_bin = i;
            }
        }
    }

    if (best_axis == INVALID_INDEX)
        return equal_partition(fallback_axis);

    const float min_center = component(center_bounds.min, best_axis);
    const float extent = component(center_bounds.size(), best_axis);
    const float split_position = min_center + extent * (float(best_bin + 1) / float(bin_count));
    auto split = std::partition(
        triangles.begin() + begin,
        triangles.begin() + end,
        [best_axis, split_position](const BuildTriangle& triangle)
        {
            return component(triangle.center, best_axis) < split_position;
        }
    );
    const uint32_t split_index = static_cast<uint32_t>(split - triangles.begin());
    return split_index == begin || split_index == end ? equal_partition(fallback_axis) : split_index;
}

uint32_t EmissiveTriangleTree::build_node(
    std::vector<BuildTriangle>& triangles,
    uint32_t begin,
    uint32_t end,
    uint32_t depth,
    const BuildOptions& options
)
{
    const uint32_t node_index = static_cast<uint32_t>(m_nodes.size());
    m_nodes.emplace_back();
    m_build_aggregates.emplace_back();
    m_stats.max_depth = std::max(m_stats.max_depth, depth);

    const uint32_t triangle_count = end - begin;
    if (triangle_count <= options.max_triangle_count_per_leaf) {
        m_build_aggregates[node_index] = compute_aggregate(std::span(triangles).subspan(begin, end - begin));
        m_nodes[node_index] = node_data_from_aggregate(m_build_aggregates[node_index]);
        shared::EmissiveTriangleTreeNode& node = m_nodes[node_index];
        node.triangle_offset = static_cast<uint32_t>(m_triangle_ids.size());
        node.triangle_count = triangle_count;
        for (uint32_t i = begin; i < end; ++i) {
            m_active_triangle_leaf_indices[triangles[i].active_index] = node_index;
            m_triangle_ids.push_back(triangles[i].id);
        }
        m_leaf_node_indices.push_back(node_index);
        ++m_stats.leaf_count;
        return node_index;
    }

    const uint32_t split_index = partition(triangles, begin, end, options);
    const uint32_t left_child = build_node(triangles, begin, split_index, depth + 1, options);
    FALCOR_CHECK(left_child == node_index + 1, "EmissiveTriangleTree preorder layout invariant failed.");
    const uint32_t right_child = build_node(triangles, split_index, end, depth + 1, options);

    // Reconstruct internal attributes from the decoded child nodes. This is the
    // same recursive aggregation contract used by GPU refitting, including the
    // conservative widening introduced by packing at every tree level.
    m_build_aggregates[node_index] = merge_aggregates(decoded_aggregate(left_child), decoded_aggregate(right_child));
    m_nodes[node_index] = node_data_from_aggregate(m_build_aggregates[node_index]);
    m_nodes[node_index].right_child = right_child;

    if (m_internal_node_indices_by_depth.size() <= depth)
        m_internal_node_indices_by_depth.resize(depth + 1);
    m_internal_node_indices_by_depth[depth].push_back(node_index);
    return node_index;
}

void EmissiveTriangleTree::build(const EmissiveGeometrySystem& emissive_geometry, const BuildOptions& options)
{
    FALCOR_CHECK(
        options.max_triangle_count_per_leaf > 0,
        "EmissiveTriangleTree leaves must contain at least one triangle."
    );
    FALCOR_CHECK(
        options.max_triangle_count_per_leaf <= shared::PackedEmissiveTriangleTreeNode::MAX_TRIANGLE_COUNT,
        "EmissiveTriangleTree supports at most {} triangles per leaf.",
        shared::PackedEmissiveTriangleTreeNode::MAX_TRIANGLE_COUNT
    );

    m_built = false;
    m_nodes.clear();
    m_build_aggregates.clear();
    m_internal_node_indices_by_depth.clear();
    m_leaf_node_indices.clear();
    m_refit_node_indices.clear();
    m_internal_refit_ranges.clear();
    m_leaf_refit_range = {};
    m_triangle_ids.clear();
    m_stats = {};

    const auto triangles = emissive_geometry.triangles();
    const auto active_triangle_ids = emissive_geometry.active_triangle_ids();
    const auto active_triangle_flux = emissive_geometry.active_triangle_flux();
    FALCOR_CHECK(
        active_triangle_ids.size() == active_triangle_flux.size(),
        "Active emissive triangle IDs and flux arrays must have equal length."
    );
    FALCOR_CHECK(
        active_triangle_ids.size() <= uint64_t(shared::PackedEmissiveTriangleTreeNode::TRIANGLE_OFFSET_MASK) + 1,
        "EmissiveTriangleTree supports at most {} active triangles.",
        uint64_t(shared::PackedEmissiveTriangleTreeNode::TRIANGLE_OFFSET_MASK) + 1
    );

    std::vector<BuildTriangle> build_triangles;
    build_triangles.reserve(active_triangle_ids.size());
    m_active_triangle_leaf_indices.assign(active_triangle_ids.size(), INVALID_INDEX);
    for (uint32_t active_index = 0; active_index < active_triangle_ids.size(); ++active_index) {
        const shared::EmissiveTriangleID id = active_triangle_ids[active_index];
        const shared::EmissiveTriangle& triangle = triangles[static_cast<uint32_t>(id)];
        AABB bounds;
        bounds.expand(triangle.pos_ws[0]);
        bounds.expand(triangle.pos_ws[1]);
        bounds.expand(triangle.pos_ws[2]);
        build_triangles.push_back({
            .id = id,
            .active_index = active_index,
            .bounds_min = bounds.min,
            .bounds_max = bounds.max,
            .center = bounds.center(),
            .normal = triangle.normal_ws,
            .flux = active_triangle_flux[active_index],
        });
    }

    if (!build_triangles.empty())
        build_node(build_triangles, 0, static_cast<uint32_t>(build_triangles.size()), 0, options);

    FALCOR_CHECK(
        m_nodes.size() < shared::PackedEmissiveTriangleTreeNode::LEAF_BIT,
        "EmissiveTriangleTree supports fewer than {} nodes.",
        shared::PackedEmissiveTriangleTreeNode::LEAF_BIT
    );

    m_stats.node_count = static_cast<uint32_t>(m_nodes.size());
    upload();
    m_built = true;
}

void EmissiveTriangleTree::create_refit_kernels()
{
    if (m_refit_leaves_kernel)
        return;

    ref<sgl::SlangModule> module = m_device->load_module("falcor2/render/kernels/emissive_triangle_tree_refit.slang");
    auto create_kernel = [&](std::string_view entry_point_name)
    {
        return m_device->create_compute_kernel({
            .program = m_device->link_program({module}, {module->entry_point(entry_point_name)}),
        });
    };
    m_refit_leaves_kernel = create_kernel("refit_leaves");
    m_refit_internal_nodes_kernel = create_kernel("refit_internal_nodes");
}

void EmissiveTriangleTree::build_refit_schedule()
{
    m_refit_node_indices.clear();
    m_internal_refit_ranges.resize(m_internal_node_indices_by_depth.size());

    for (uint32_t depth = 0; depth < m_internal_node_indices_by_depth.size(); ++depth) {
        const std::vector<uint32_t>& indices = m_internal_node_indices_by_depth[depth];
        m_internal_refit_ranges[depth] = {
            .offset = static_cast<uint32_t>(m_refit_node_indices.size()),
            .count = static_cast<uint32_t>(indices.size()),
        };
        m_refit_node_indices.insert(m_refit_node_indices.end(), indices.begin(), indices.end());
    }

    m_leaf_refit_range = {
        .offset = static_cast<uint32_t>(m_refit_node_indices.size()),
        .count = static_cast<uint32_t>(m_leaf_node_indices.size()),
    };
    m_refit_node_indices.insert(m_refit_node_indices.end(), m_leaf_node_indices.begin(), m_leaf_node_indices.end());
    FALCOR_CHECK_EQ(m_refit_node_indices.size(), m_nodes.size());

    m_refit_dispatch_count = m_leaf_refit_range.count > 0 ? 1 : 0;
    for (const RefitRange& range : m_internal_refit_ranges)
        m_refit_dispatch_count += range.count > 0 ? 1 : 0;
}

void EmissiveTriangleTree::refit(const EmissiveGeometrySystem& emissive_geometry, sgl::CommandEncoder* command_encoder)
{
    FALCOR_CHECK(m_built, "Cannot refit an EmissiveTriangleTree before it has been built.");
    FALCOR_CHECK(
        emissive_geometry.active_triangle_count() == m_triangle_ids.size(),
        "EmissiveTriangleTree topology changed; rebuild instead of refitting."
    );
    if (m_stats.node_count == 0)
        return;

    FALCOR_CHECK_NOT_NULL(m_nodes_buffer);
    FALCOR_CHECK_NOT_NULL(m_refit_weighted_normal_sums_buffer);
    FALCOR_CHECK_NOT_NULL(m_refit_node_indices_buffer);
    FALCOR_CHECK_NOT_NULL(m_triangle_ids_buffer);
    create_refit_kernels();

    ref<sgl::CommandEncoder> owned_command_encoder;
    if (!command_encoder) {
        owned_command_encoder = m_device->create_command_encoder();
        command_encoder = owned_command_encoder.get();
    }

    auto bind_refit_resources = [&](sgl::ShaderCursor cursor, const RefitRange& range)
    {
        cursor = cursor.find_entry_point(0);
        cursor["nodes"] = m_nodes_buffer;
        cursor["weighted_normal_sums"] = m_refit_weighted_normal_sums_buffer;
        cursor["node_indices"] = m_refit_node_indices_buffer;
        cursor["first_node_offset"] = range.offset;
        cursor["node_count"] = range.count;
        return cursor;
    };

    m_refit_leaves_kernel->dispatch(
        uint3(m_leaf_refit_range.count, 1, 1),
        [&](sgl::ShaderCursor cursor)
        {
            cursor = bind_refit_resources(cursor, m_leaf_refit_range);
            emissive_geometry.bind_to_scene(cursor);
            cursor["triangle_ids"] = m_triangle_ids_buffer;
        },
        command_encoder
    );
    command_encoder->global_barrier();

    for (size_t depth = m_internal_refit_ranges.size(); depth > 0; --depth) {
        const RefitRange range = m_internal_refit_ranges[depth - 1];
        if (range.count == 0)
            continue;
        m_refit_internal_nodes_kernel->dispatch(
            uint3(range.count, 1, 1),
            [&](sgl::ShaderCursor cursor)
            {
                bind_refit_resources(cursor, range);
            },
            command_encoder
        );
        command_encoder->global_barrier();
    }

    if (owned_command_encoder)
        m_device->submit_command_buffer(owned_command_encoder->finish());
}

void EmissiveTriangleTree::upload_nodes()
{
    std::vector<shared::PackedEmissiveTriangleTreeNode> packed_nodes;
    packed_nodes.reserve(m_nodes.size());
    for (const shared::EmissiveTriangleTreeNode& node : m_nodes) {
        const shared::PackedEmissiveTriangleTreeNode packed = shared::detail::pack_emissive_triangle_tree_node(node);
        const shared::EmissiveTriangleTreeNode unpacked = shared::detail::unpack_emissive_triangle_tree_node(packed);
        const float3 node_bounds_min = node.get_bounds_min();
        const float3 node_bounds_max = node.get_bounds_max();
        const float3 unpacked_bounds_min = unpacked.get_bounds_min();
        const float3 unpacked_bounds_max = unpacked.get_bounds_max();
        for (uint32_t axis = 0; axis < 3; ++axis) {
            FALCOR_CHECK(
                std::isfinite(unpacked_bounds_min[axis]) && std::isfinite(unpacked_bounds_max[axis])
                    && unpacked_bounds_min[axis] <= node_bounds_min[axis]
                    && unpacked_bounds_max[axis] >= node_bounds_max[axis],
                "EmissiveTriangleTree node bounds exceed the supported packed coordinate range."
            );
        }
        FALCOR_CHECK(
            std::isfinite(unpacked.flux) && std::isfinite(unpacked.cos_cone_angle)
                && std::isfinite(unpacked.cone_direction.x) && std::isfinite(unpacked.cone_direction.y)
                && std::isfinite(unpacked.cone_direction.z),
            "EmissiveTriangleTree node attributes must remain finite after packing."
        );
        packed_nodes.push_back(packed);
    }

    shared::EmissiveTriangleTreeNode empty_node_data{};
    empty_node_data.right_child = INVALID_INDEX;
    const shared::PackedEmissiveTriangleTreeNode empty_node
        = shared::detail::pack_emissive_triangle_tree_node(empty_node_data);

    const void* nodes_data = packed_nodes.empty() ? static_cast<const void*>(&empty_node) : packed_nodes.data();
    const size_t nodes_size = packed_nodes.empty() ? sizeof(empty_node) : packed_nodes.size() * sizeof(packed_nodes[0]);
    if (!m_nodes_buffer || m_nodes_buffer->size() != nodes_size) {
        m_nodes_buffer = m_device->create_buffer({
            .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
            .label = "EmissiveTriangleTree::nodes",
            .data = nodes_data,
            .data_size = nodes_size,
        });
        ++m_node_buffer_recreation_count;
    } else {
        m_nodes_buffer->set_data(nodes_data, nodes_size);
    }
}

void EmissiveTriangleTree::upload()
{
    upload_nodes();
    build_refit_schedule();

    const uint32_t invalid_index = INVALID_INDEX;

    std::vector<float4> weighted_normal_sums;
    weighted_normal_sums.reserve(m_build_aggregates.size());
    for (const BuildAggregate& aggregate : m_build_aggregates)
        weighted_normal_sums.emplace_back(aggregate.weighted_normal_sum, 0.f);
    const float4 zero_weighted_normal_sum(0.f);
    m_refit_weighted_normal_sums_buffer = m_device->create_buffer({
        .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
        .label = "EmissiveTriangleTree::refit_weighted_normal_sums",
        .data = weighted_normal_sums.empty() ? static_cast<const void*>(&zero_weighted_normal_sum)
                                             : weighted_normal_sums.data(),
        .data_size = weighted_normal_sums.empty() ? sizeof(zero_weighted_normal_sum)
                                                  : weighted_normal_sums.size() * sizeof(weighted_normal_sums[0]),
    });

    m_refit_node_indices_buffer = m_device->create_buffer({
        .usage = sgl::BufferUsage::shader_resource,
        .label = "EmissiveTriangleTree::refit_node_indices",
        .data = m_refit_node_indices.empty() ? static_cast<const void*>(&invalid_index) : m_refit_node_indices.data(),
        .data_size = m_refit_node_indices.empty() ? sizeof(invalid_index)
                                                  : m_refit_node_indices.size() * sizeof(m_refit_node_indices[0]),
    });

    const void* triangle_ids_data
        = m_triangle_ids.empty() ? static_cast<const void*>(&invalid_index) : m_triangle_ids.data();
    const size_t triangle_ids_size
        = m_triangle_ids.empty() ? sizeof(invalid_index) : m_triangle_ids.size() * sizeof(m_triangle_ids[0]);
    m_triangle_ids_buffer = m_device->create_buffer({
        .usage = sgl::BufferUsage::shader_resource,
        .label = "EmissiveTriangleTree::triangle_ids",
        .data = triangle_ids_data,
        .data_size = triangle_ids_size,
    });

    const void* leaf_indices_data = m_active_triangle_leaf_indices.empty() ? static_cast<const void*>(&invalid_index)
                                                                           : m_active_triangle_leaf_indices.data();
    const size_t leaf_indices_size = m_active_triangle_leaf_indices.empty()
        ? sizeof(invalid_index)
        : m_active_triangle_leaf_indices.size() * sizeof(m_active_triangle_leaf_indices[0]);
    m_active_triangle_leaf_indices_buffer = m_device->create_buffer({
        .usage = sgl::BufferUsage::shader_resource,
        .label = "EmissiveTriangleTree::active_triangle_leaf_indices",
        .data = leaf_indices_data,
        .data_size = leaf_indices_size,
    });
}

void EmissiveTriangleTree::write_to_cursor(sgl::ShaderCursor cursor) const
{
    cursor["nodes"] = m_nodes_buffer;
    cursor["triangle_ids"] = m_triangle_ids_buffer;
    cursor["active_triangle_leaf_indices"] = m_active_triangle_leaf_indices_buffer;
}

} // namespace falcor
