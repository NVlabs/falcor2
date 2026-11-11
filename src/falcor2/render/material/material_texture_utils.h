// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/texture_manager.h"
#include <sgl/device/formats.h>
#include <sgl/device/sampler.h>

namespace falcor {

// Encoded normal maps store XY and require positive Z reconstruction.
inline bool is_normal_map_encoded(const TextureHandle& normal_map)
{
    const auto texture = normal_map.texture();
    return texture && sgl::get_format_info(texture->format()).channel_count < 3;
}

// Explicit resources take precedence over paths. Keep deferred loading and
// empty-binding reset identical for all surface-material frontends.
inline TextureHandle load_material_texture(
    TextureManager* manager,
    const ref<sgl::Texture>& texture,
    const ref<sgl::Sampler>& sampler,
    const std::filesystem::path& path,
    TextureLoadFlags flags
)
{
    if (texture)
        return manager->register_texture({.texture = texture, .sampler = sampler});
    if (!path.empty())
        return manager->load_texture({.path = path, .sampler = sampler, .flags = flags, .load_deferred = true});
    return {};
}

} // namespace falcor
