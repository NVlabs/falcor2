// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"
#include "falcor2/core/macros.h"
#include "shared_gltf_texture_selectors.h"

#include <compare>
#include <cstdint>

namespace falcor {

/// Selects how many native glTF shader types a scene uses within each material workflow.
enum class GLTFSpecializationMode {
    /// Compile a type for each material's effective masks.
    required,
    /// Compile one type for the live union of each workflow's effective masks.
    single,
    /// Compile one type with every allowed capability for each workflow.
    all,
    /// Build a small number of material specializations to cover all requirements.
    balanced,
};
SGL_ENUM_INFO(
    GLTFSpecializationMode,
    {
        {GLTFSpecializationMode::required, "required"},
        {GLTFSpecializationMode::single, "single"},
        {GLTFSpecializationMode::all, "all"},
        {GLTFSpecializationMode::balanced, "balanced"},
    }
);
SGL_ENUM_REGISTER(GLTFSpecializationMode);

/// Feature bits that native glTF materials may use at runtime and in shader specializations.
enum class GLTFFeatureFlags : uint32_t {
    none = 0,
    normal = shared::gltfmat::FeatureBits::NORMAL,
    emission = shared::gltfmat::FeatureBits::EMISSION,
    vertex_color = shared::gltfmat::FeatureBits::VERTEX_COLOR,
    multiscatter = shared::gltfmat::FeatureBits::MULTISCATTER,
    specular = shared::gltfmat::FeatureBits::SPECULAR,
    transmission = shared::gltfmat::FeatureBits::TRANSMISSION,
    ior = shared::gltfmat::FeatureBits::IOR,
    volume = shared::gltfmat::FeatureBits::VOLUME,
    clearcoat = shared::gltfmat::FeatureBits::CLEARCOAT,
    sheen = shared::gltfmat::FeatureBits::SHEEN,
    iridescence = shared::gltfmat::FeatureBits::IRIDESCENCE,
    anisotropy = shared::gltfmat::FeatureBits::ANISOTROPY,
    diffuse_transmission = shared::gltfmat::FeatureBits::DIFFUSE_TRANSMISSION,
    all = shared::gltfmat::FeatureBits::ALL_MR,
};
FALCOR_ENUM_CLASS_OPERATORS(GLTFFeatureFlags);
SGL_ENUM_FLAGS_INFO(
    GLTFFeatureFlags,
    {
        {GLTFFeatureFlags::none, "none"},
        {GLTFFeatureFlags::normal, "normal"},
        {GLTFFeatureFlags::emission, "emission"},
        {GLTFFeatureFlags::vertex_color, "vertex_color"},
        {GLTFFeatureFlags::multiscatter, "multiscatter"},
        {GLTFFeatureFlags::specular, "specular"},
        {GLTFFeatureFlags::transmission, "transmission"},
        {GLTFFeatureFlags::ior, "ior"},
        {GLTFFeatureFlags::volume, "volume"},
        {GLTFFeatureFlags::clearcoat, "clearcoat"},
        {GLTFFeatureFlags::sheen, "sheen"},
        {GLTFFeatureFlags::iridescence, "iridescence"},
        {GLTFFeatureFlags::anisotropy, "anisotropy"},
        {GLTFFeatureFlags::diffuse_transmission, "diffuse_transmission"},
        {GLTFFeatureFlags::all, "all"},
    }
);
SGL_ENUM_REGISTER(GLTFFeatureFlags);

/// Four shader specialization masks. Runtime masks must be contained in their selected compile masks.
struct GLTFMasks {
    uint32_t features{0};
    uint32_t textures{0};
    uint32_t transforms{0};
    uint32_t nonzero_texcoords{0};

    GLTFMasks merged(const GLTFMasks& other) const
    {
        return {
            features | other.features,
            textures | other.textures,
            transforms | other.transforms,
            nonzero_texcoords | other.nonzero_texcoords,
        };
    }

    bool covers(const GLTFMasks& required) const
    {
        return (features & required.features) == required.features
            && (textures & required.textures) == required.textures
            && (transforms & required.transforms) == required.transforms
            && (nonzero_texcoords & required.nonzero_texcoords) == required.nonzero_texcoords;
    }

    auto operator<=>(const GLTFMasks&) const = default;
};

} // namespace falcor
