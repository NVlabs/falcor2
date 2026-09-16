// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/shared_emissive_triangle_tree_types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>

using namespace falcor;

namespace {

void check_bounds_containment(const float3 bounds_min, const float3 bounds_max)
{
    shared::EmissiveTriangleTreeNode data{};
    data.set_bounds(bounds_min, bounds_max);
    data.flux = 123.25f;
    data.right_child = 17;
    data.cone_direction = normalize(float3(1.f, -2.f, 3.f));
    data.cos_cone_angle = 0.25f;

    const shared::PackedEmissiveTriangleTreeNode packed = shared::detail::pack_emissive_triangle_tree_node(data);
    const shared::EmissiveTriangleTreeNode unpacked = shared::detail::unpack_emissive_triangle_tree_node(packed);
    const float3 unpacked_bounds_min = unpacked.get_bounds_min();
    const float3 unpacked_bounds_max = unpacked.get_bounds_max();

    for (uint32_t axis = 0; axis < 3; ++axis) {
        CHECK(std::isfinite(unpacked_bounds_min[axis]));
        CHECK(std::isfinite(unpacked_bounds_max[axis]));
        CHECK_LE(unpacked_bounds_min[axis], bounds_min[axis]);
        CHECK_GE(unpacked_bounds_max[axis], bounds_max[axis]);
    }
    CHECK_EQ(std::bit_cast<uint32_t>(unpacked.flux), std::bit_cast<uint32_t>(data.flux));

    const float source_angle = std::acos(data.cos_cone_angle);
    const float direction_error = std::acos(std::clamp(dot(data.cone_direction, unpacked.cone_direction), -1.f, 1.f));
    const float unpacked_angle = std::acos(unpacked.cos_cone_angle);
    CHECK_GE(unpacked_angle + 1e-6f, source_angle + direction_error);
}

} // namespace

TEST_CASE("EmissiveTriangleTree packed node is conservative")
{
    check_bounds_containment(float3(-1.f, -2.f, -3.f), float3(4.f, 5.f, 6.f));
    check_bounds_containment(float3(1e20f, -1e20f, 1e10f), float3(1e20f + 1e15f, -1e20f + 2e15f, 1e10f + 1e5f));
    check_bounds_containment(float3(-1e30f, -2e30f, -3e30f), float3(-0.9e30f, -1.8e30f, -2.7e30f));

    shared::EmissiveTriangleTreeNode data{};
    data.set_bounds(float3(-2.086941f, -1.279461f, -2.586075f), float3(1.705780f, 1.289739f, 1.931448f));
    data.flux = 1.f;
    data.right_child = 1;
    data.cone_direction = float3(0.f, 0.f, 1.f);
    data.cos_cone_angle = 1.f;
    for (uint32_t iteration = 0; iteration < 32; ++iteration) {
        const shared::EmissiveTriangleTreeNode previous = data;
        data = shared::detail::unpack_emissive_triangle_tree_node(
            shared::detail::pack_emissive_triangle_tree_node(data)
        );
        const float3 previous_bounds_min = previous.get_bounds_min();
        const float3 previous_bounds_max = previous.get_bounds_max();
        const float3 data_bounds_min = data.get_bounds_min();
        const float3 data_bounds_max = data.get_bounds_max();
        for (uint32_t axis = 0; axis < 3; ++axis) {
            CHECK_LE(data_bounds_min[axis], previous_bounds_min[axis]);
            CHECK_GE(data_bounds_max[axis], previous_bounds_max[axis]);
        }
    }
}

TEST_CASE("EmissiveTriangleTree cone cosine packing rounds outward")
{
    const float reported_boundary_case = std::bit_cast<float>(0x3dccd8ccu);
    CHECK_LE(
        shared::detail::decode_emissive_triangle_tree_cone_cos(
            shared::detail::encode_emissive_triangle_tree_cone_cos(reported_boundary_case)
        ),
        reported_boundary_case
    );

    for (uint32_t code = 1; code <= 0xffff; ++code) {
        const float quantization_boundary = shared::detail::decode_emissive_triangle_tree_cone_cos(code);
        const float source_cosine = std::nextafter(quantization_boundary, -std::numeric_limits<float>::infinity());
        const uint32_t encoded = shared::detail::encode_emissive_triangle_tree_cone_cos(source_cosine);
        const float decoded = shared::detail::decode_emissive_triangle_tree_cone_cos(encoded);
        CHECK_LE(decoded, source_cosine);
    }
}

TEST_CASE("EmissiveTriangleTree packed metadata round-trips supported limits")
{
    shared::EmissiveTriangleTreeNode leaf{};
    leaf.set_bounds(float3(-1.f), float3(1.f));
    leaf.flux = 1.f;
    leaf.cone_direction = float3(0.f, 0.f, 1.f);
    leaf.cos_cone_angle = 1.f;
    leaf.triangle_offset = shared::PackedEmissiveTriangleTreeNode::TRIANGLE_OFFSET_MASK;
    leaf.triangle_count = shared::PackedEmissiveTriangleTreeNode::MAX_TRIANGLE_COUNT;

    const shared::PackedEmissiveTriangleTreeNode packed_leaf = shared::detail::pack_emissive_triangle_tree_node(leaf);
    CHECK(packed_leaf.is_leaf());
    CHECK_EQ(packed_leaf.triangle_offset(), leaf.triangle_offset);
    CHECK_EQ(packed_leaf.triangle_count(), leaf.triangle_count);

    shared::EmissiveTriangleTreeNode internal = leaf;
    internal.triangle_count = 0;
    internal.right_child = shared::PackedEmissiveTriangleTreeNode::LEAF_BIT - 1;
    const shared::PackedEmissiveTriangleTreeNode packed_internal
        = shared::detail::pack_emissive_triangle_tree_node(internal);
    CHECK_FALSE(packed_internal.is_leaf());
    CHECK_EQ(packed_internal.right_child(), internal.right_child);
}
