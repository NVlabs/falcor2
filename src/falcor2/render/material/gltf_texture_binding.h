// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gltf_texture_transform.h"
#include "material_texture_utils.h"

#include <filesystem>
#include <functional>
#include <type_traits>
#include <utility>

namespace falcor {

/// Authored and resolved state for one glTF material texture slot. This is CPU
/// storage; the GPU texture-info layout is defined separately in Slang.
struct GLTFTextureBinding {
    ref<sgl::Texture> texture;
    ref<sgl::Sampler> sampler;
    ref<Blob> bytes;
    std::filesystem::path path;
    uint32_t effective_texcoord{0};
    GLTFTextureTransform transform;
    TextureHandle handle;

    /// Resolve the authored object, bytes, or path using the existing material precedence.
    void load(TextureManager* manager, TextureLoadFlags flags)
    {
        handle = load_material_texture(manager, texture, sampler, bytes, path, flags);
    }
};

/// Visit bindings in the workflow's fixed slot order without copying their transforms.
/// binding_at(index) must return a reference to a GLTFTextureBinding.
template<typename BindingAt>
inline uint32_t gltf_texture_binding_transform_mask(size_t binding_count, BindingAt&& binding_at)
{
    static_assert(std::is_lvalue_reference_v<std::invoke_result_t<BindingAt&, size_t>>);
    return gltf_texture_transform_mask(
        binding_count,
        [&](size_t index) -> const GLTFTextureTransform&
        {
            return std::invoke(binding_at, index).transform;
        }
    );
}

/// Write active transform rows in the same fixed slot order as the mask.
template<typename BindingAt>
inline void write_gltf_texture_binding_transforms(
    sgl::BufferCursor& cursor,
    size_t binding_count,
    BindingAt&& binding_at,
    uint32_t active_mask,
    UVOrigin scene_uv_origin
)
{
    static_assert(std::is_lvalue_reference_v<std::invoke_result_t<BindingAt&, size_t>>);
    write_gltf_texture_transforms(
        cursor,
        binding_count,
        [&](size_t index) -> const GLTFTextureTransform&
        {
            return std::invoke(binding_at, index).transform;
        },
        active_mask,
        scene_uv_origin
    );
}

template<typename BindingAt>
inline void write_gltf_texture_binding_transforms(
    sgl::BufferCursor& cursor,
    size_t binding_count,
    BindingAt&& binding_at,
    UVOrigin scene_uv_origin
)
{
    const uint32_t active_mask = gltf_texture_binding_transform_mask(binding_count, binding_at);
    write_gltf_texture_binding_transforms(
        cursor,
        binding_count,
        std::forward<BindingAt>(binding_at),
        active_mask,
        scene_uv_origin
    );
}

} // namespace falcor
