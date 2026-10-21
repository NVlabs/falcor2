// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/importer_types.h"
#include "falcor2/render/texture_load_flags.h"

#include <functional>
#include <optional>
#include <span>
#include <string_view>

namespace falcor {

using GLTFTextureSetter = std::function<void(Properties&, std::string_view, const ImporterTexture&, TextureLoadFlags)>;

/// Approximate compatibility conversion. Unsupported authored features are logged
/// and listed in _gltf_conversion_warnings; no GPU features are added to the target.
FALCOR_API std::optional<Properties> gltf_to_standard(
    const ImporterMaterial& source,
    std::span<const ImporterTexture> textures,
    bool use_vertex_color,
    const GLTFTextureSetter& set_texture
);

/// Map existing OpenPBR controls. Texture transforms and SpecGloss are unsupported.
/// SpecGloss throws instead of silently switching target; other limitations are reported.
FALCOR_API std::optional<Properties> gltf_to_openpbr(
    const ImporterMaterial& source,
    std::span<const ImporterTexture> textures,
    bool use_vertex_color,
    const GLTFTextureSetter& set_texture
);

/// Generate a MaterialX glTF graph. resolve_texture returns a readable filename,
/// including for embedded resources. UV transforms are emitted in authored glTF
/// coordinates; uv_origin describes the live scene's texture-coordinate convention.
FALCOR_API std::optional<Properties> gltf_to_mtlx(
    const ImporterMaterial& source,
    std::span<const ImporterTexture> textures,
    bool use_vertex_color,
    UVOrigin uv_origin,
    const std::function<std::filesystem::path(const ImporterTexture&)>& resolve_texture
);

/// Convert native glTF inputs without creating a scene material. Returns no value
/// for an unrelated source type. The callback binds each texture to the named
/// destination property (including its path/sampler); resource ownership stays
/// with the caller. Invalid texture indices or UV selectors throw.
FALCOR_API std::optional<Properties> gltf_to_gltf(
    const ImporterMaterial& source,
    std::span<const ImporterTexture> textures,
    bool use_vertex_color,
    const std::function<void(Properties&, std::string_view, const ImporterTexture&, TextureLoadFlags)>& set_texture
);

/// Generate properties for a class-compiled SDK glTF wrapper without creating a
/// material. Returns no value for sources other than glTF MR/SpecGloss. The callback
/// binds MDL texture arguments; the bool selects color versus data decoding.
/// UV selectors and transforms remain per-material arguments.
/// uv_origin describes the live scene's tangent/texture-coordinate convention.
FALCOR_API std::optional<Properties> gltf_to_mdl(
    const ImporterMaterial& source,
    std::span<const ImporterTexture> textures,
    UVOrigin uv_origin,
    const std::function<void(Properties&, std::string_view, const ImporterTexture&, bool)>& set_texture
);

} // namespace falcor
