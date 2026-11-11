// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/error.h"
#include "falcor2/core/types.h"
#include "falcor2/render/scene_types.h"
#include <sgl/device/buffer_cursor.h>
#include <cmath>
#include <span>

namespace falcor {

// Stored in glTF's authored upper-left UV convention. Scene UV conversion happens
// only when uploading the transform rows; these values remain unchanged.
struct GLTFTextureTransform {
    float2 offset{0.f};
    float rotation{0.f};
    float2 scale{1.f};

    bool is_identity() const
    {
        return offset.x == 0.f && offset.y == 0.f && rotation == 0.f && scale.x == 1.f && scale.y == 1.f;
    }
};

inline uint32_t gltf_texture_transform_mask(std::span<const GLTFTextureTransform> transforms)
{
    FALCOR_CHECK(transforms.size() <= 32, "Too many glTF texture bindings for the transform mask.");
    uint32_t mask = 0;
    for (size_t i = 0; i < transforms.size(); ++i)
        if (!transforms[i].is_identity())
            mask |= 1u << i;
    return mask;
}

// Writes only active records, in binding order. The caller owns the shared Material allocation.
// Converts authored upper-left transforms to scene_uv_origin for upload without modifying transforms.
inline void write_gltf_texture_transforms(
    sgl::BufferCursor& cursor,
    std::span<const GLTFTextureTransform> transforms,
    UVOrigin scene_uv_origin
)
{
    uint32_t index = 0;
    for (const auto& transform : transforms) {
        if (transform.is_identity())
            continue;
        const float c = std::cos(transform.rotation);
        const float s = std::sin(transform.rotation);
        // Match glTF's Y-down rotation and Khronos Sample Renderer. The README's
        // GLSL matrix example has the opposite sign (glTF issue #1563).
        float3 row_u{c * transform.scale.x, s * transform.scale.y, transform.offset.x};
        float3 row_v{-s * transform.scale.x, c * transform.scale.y, transform.offset.y};
        if (scene_uv_origin == UVOrigin::lower_left) {
            // Conjugate the authored transform by (u,v) -> (u,1-v).
            row_u.z += row_u.y;
            row_u.y = -row_u.y;
            row_v = float3(-row_v.x, row_v.y, 1.f - row_v.y - row_v.z);
        }
        cursor[index]["row_u"] = row_u;
        cursor[index]["row_v"] = row_v;
        ++index;
    }
    cursor.apply();
}

} // namespace falcor
