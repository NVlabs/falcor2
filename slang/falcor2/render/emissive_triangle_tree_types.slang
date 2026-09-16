// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FALCOR_SHARED_HOST
implementing render;
#define FALCOR_EMISSIVE_TREE_CONST
#define FALCOR_EMISSIVE_TREE_MUTATING [mutating]
#define FALCOR_EMISSIVE_TREE_STATIC_CONST static const
#else
#define FALCOR_EMISSIVE_TREE_CONST const
#define FALCOR_EMISSIVE_TREE_MUTATING
#define FALCOR_EMISSIVE_TREE_STATIC_CONST inline static constexpr
#endif

namespace detail {

/// Encodes an extent so reconstructing center +/- extent conservatively contains the source bounds.
/// Verify the decoded endpoints because floating-point endpoint arithmetic can still round inward.
public static uint encode_emissive_triangle_tree_bound_extent(float center, float bounds_min, float bounds_max)
{
    uint encoded = pack_ufloat_e8m8_round_up(max(center - bounds_min, bounds_max - center));
    const float decoded = unpack_ufloat_e8m8(encoded);
    if (center - decoded > bounds_min || center + decoded < bounds_max)
        encoded = encoded < 0xfeff ? encoded + 1 : 0xffff;
    return encoded;
}

public static float decode_emissive_triangle_tree_cone_cos(uint encoded)
{
    return float(encoded) * (2.0f / 65535.0f) - 1.0f;
}

/// Encodes a cone cosine so the decoded value is no greater than the source, preserving or widening the cone.
/// Verify the decoded cosine because the floating-point scale can round up before conversion to uint truncates.
public static uint encode_emissive_triangle_tree_cone_cos(float cone_cos)
{
    cone_cos = clamp(cone_cos, -1.0f, 1.0f);
    uint encoded = uint(saturate(0.5f * cone_cos + 0.5f) * 65535.0f);
    if (encoded > 0 && decode_emissive_triangle_tree_cone_cos(encoded) > cone_cos)
        --encoded;
    return encoded;
}

public static float emissive_triangle_tree_sin_from_cos(float cos_angle)
{
#ifdef FALCOR_SHARED_HOST
    return std::sqrt(max(0.0f, 1.0f - cos_angle * cos_angle));
#else
    return sqrt(max(0.0f, 1.0f - cos_angle * cos_angle));
#endif
}

/// Returns the cosine of a cone widened by the angular difference between its
/// direction and a new direction. A result of -1 marks a full-sphere cone.
public static float widen_emissive_triangle_tree_cone_cos(float cone_cos, float direction_difference_cos)
{
    cone_cos = clamp(cone_cos, -1.0f, 1.0f);
    direction_difference_cos = clamp(direction_difference_cos, -1.0f, 1.0f);
    if (cone_cos <= -direction_difference_cos)
        return -1.0f;

    const float cone_sin = emissive_triangle_tree_sin_from_cos(cone_cos);
    const float direction_difference_sin = emissive_triangle_tree_sin_from_cos(direction_difference_cos);
    return max(-1.0f, cone_cos * direction_difference_cos - cone_sin * direction_difference_sin);
}

} // namespace detail

/// Controls how emissive triangles are selected within a tree leaf.
public enum class EmissiveTriangleTreeLeafSamplingMode : uint {
    uniform,
    contextual_importance,
};
#ifdef FALCOR_SHARED_HOST
SGL_ENUM_INFO(
    EmissiveTriangleTreeLeafSamplingMode,
    {
        { EmissiveTriangleTreeLeafSamplingMode::uniform, "uniform" },
        { EmissiveTriangleTreeLeafSamplingMode::contextual_importance, "contextual_importance" },
    }
);
SGL_ENUM_REGISTER(EmissiveTriangleTreeLeafSamplingMode);
#endif

/// Unpacked emissive triangle tree node data used while constructing and updating the tree.
public struct EmissiveTriangleTreeNode
{
    public float3 center;
    public float flux;

    public float3 extent;
    public uint right_child;

    public float3 cone_direction;
    public float cos_cone_angle;

    public uint triangle_offset;
    public uint triangle_count;

    public bool is_leaf() FALCOR_EMISSIVE_TREE_CONST { return triangle_count != 0; }

    public float3 get_bounds_min() FALCOR_EMISSIVE_TREE_CONST { return center - extent; }

    public float3 get_bounds_max() FALCOR_EMISSIVE_TREE_CONST { return center + extent; }

    /// Stores a conservative center/extent representation of the source bounds.
    /// The extent is made exactly representable by the packed node format so
    /// subsequent pack/unpack operations are geometrically idempotent.
    FALCOR_EMISSIVE_TREE_MUTATING public void set_bounds(float3 bounds_min, float3 bounds_max)
    {
        center = 0.5f * bounds_min + 0.5f * bounds_max;
        extent = float3(
            unpack_ufloat_e8m8(
                detail::encode_emissive_triangle_tree_bound_extent(center.x, bounds_min.x, bounds_max.x)
            ),
            unpack_ufloat_e8m8(
                detail::encode_emissive_triangle_tree_bound_extent(center.y, bounds_min.y, bounds_max.y)
            ),
            unpack_ufloat_e8m8(detail::encode_emissive_triangle_tree_bound_extent(center.z, bounds_min.z, bounds_max.z))
        );
    }
};

/// Binary emissive triangle tree node packed into 32 bytes for traversal.
///
/// Nodes are stored in preorder. Internal nodes store their right child in the
/// metadata word and have their left child immediately after them. Leaves use
/// the metadata word for a type bit, a 25-bit triangle offset, and a six-bit
/// triangle count minus one.
public struct PackedEmissiveTriangleTreeNode
{
    public FALCOR_EMISSIVE_TREE_STATIC_CONST uint LEAF_BIT = 0x80000000;
    public FALCOR_EMISSIVE_TREE_STATIC_CONST uint TRIANGLE_OFFSET_BITS = 25;
    public FALCOR_EMISSIVE_TREE_STATIC_CONST uint TRIANGLE_OFFSET_MASK = (1 << TRIANGLE_OFFSET_BITS) - 1;
    public FALCOR_EMISSIVE_TREE_STATIC_CONST uint TRIANGLE_COUNT_MASK = 0x3f;
    public FALCOR_EMISSIVE_TREE_STATIC_CONST uint MAX_TRIANGLE_COUNT = TRIANGLE_COUNT_MASK + 1;

    public uint metadata;
    public float center_x;
    public float center_y;
    public float center_z;

    public uint packed_extent_xy;
    public uint packed_extent_z_cos_cone_angle;
    public uint packed_cone_direction;
    public float flux;

    public bool is_leaf() FALCOR_EMISSIVE_TREE_CONST { return (metadata & LEAF_BIT) != 0; }

    public uint right_child() FALCOR_EMISSIVE_TREE_CONST { return metadata; }

    public uint triangle_offset() FALCOR_EMISSIVE_TREE_CONST { return metadata & TRIANGLE_OFFSET_MASK; }

    public uint triangle_count() FALCOR_EMISSIVE_TREE_CONST
    {
        return ((metadata >> TRIANGLE_OFFSET_BITS) & TRIANGLE_COUNT_MASK) + 1;
    }
};

namespace detail {

public static PackedEmissiveTriangleTreeNode pack_emissive_triangle_tree_node(EmissiveTriangleTreeNode data)
{
    PackedEmissiveTriangleTreeNode node;
    node.metadata = data.is_leaf() ? PackedEmissiveTriangleTreeNode::LEAF_BIT
            | ((data.triangle_count - 1) << PackedEmissiveTriangleTreeNode::TRIANGLE_OFFSET_BITS) | data.triangle_offset
                                   : data.right_child;

    node.center_x = data.center.x;
    node.center_y = data.center.y;
    node.center_z = data.center.z;
    const float3 bounds_min = data.get_bounds_min();
    const float3 bounds_max = data.get_bounds_max();
    const uint extent_x = encode_emissive_triangle_tree_bound_extent(data.center.x, bounds_min.x, bounds_max.x);
    const uint extent_y = encode_emissive_triangle_tree_bound_extent(data.center.y, bounds_min.y, bounds_max.y);
    const uint extent_z = encode_emissive_triangle_tree_bound_extent(data.center.z, bounds_min.z, bounds_max.z);
    node.packed_extent_xy = extent_x | (extent_y << 16);

    float3 cone_direction = data.cone_direction;
    float cos_cone_angle = clamp(data.cos_cone_angle, -1.0f, 1.0f);
    if (cos_cone_angle <= -1.0f || dot(cone_direction, cone_direction) <= 0.0f) {
        cone_direction = float3(0.0f, 0.0f, 1.0f);
        cos_cone_angle = -1.0f;
    } else {
        cone_direction = normalize(cone_direction);
    }
    node.packed_cone_direction = pack_normal_oct_snorm2x16(cone_direction);
    const float3 decoded_cone_direction = unpack_normal_oct_snorm2x16(node.packed_cone_direction);
    const float widened_cos_cone_angle
        = widen_emissive_triangle_tree_cone_cos(cos_cone_angle, dot(cone_direction, decoded_cone_direction));
    const uint packed_cos_cone_angle = encode_emissive_triangle_tree_cone_cos(widened_cos_cone_angle);
    node.packed_extent_z_cos_cone_angle = extent_z | (packed_cos_cone_angle << 16);
    node.flux = data.flux;
    return node;
}

public static EmissiveTriangleTreeNode unpack_emissive_triangle_tree_node(PackedEmissiveTriangleTreeNode node)
{
    EmissiveTriangleTreeNode data;
    const float3 center = float3(node.center_x, node.center_y, node.center_z);
    const float3 extent = float3(
        unpack_ufloat_e8m8(node.packed_extent_xy & 0xffff),
        unpack_ufloat_e8m8(node.packed_extent_xy >> 16),
        unpack_ufloat_e8m8(node.packed_extent_z_cos_cone_angle & 0xffff)
    );
    data.center = center;
    data.flux = node.flux;
    data.extent = extent;
    data.right_child = node.is_leaf() ? 0xffffffff : node.right_child();
    data.cone_direction = unpack_normal_oct_snorm2x16(node.packed_cone_direction);
    data.cos_cone_angle = decode_emissive_triangle_tree_cone_cos(node.packed_extent_z_cos_cone_angle >> 16);
    data.triangle_offset = node.is_leaf() ? node.triangle_offset() : 0;
    data.triangle_count = node.is_leaf() ? node.triangle_count() : 0;
    return data;
}

} // namespace detail

#undef FALCOR_EMISSIVE_TREE_CONST
#undef FALCOR_EMISSIVE_TREE_MUTATING
#undef FALCOR_EMISSIVE_TREE_STATIC_CONST
