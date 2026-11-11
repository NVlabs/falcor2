// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"
#include <sgl/utils/texture_loader.h>

namespace falcor {

/// Optional texture transformations and common presets. Combine flags with bitwise OR;
/// presets with equal values share cached textures.
///
/// srgb explicitly interprets supported 8-bit color inputs as sRGB regardless of
/// bitmap metadata. Floating-point/HDR data and alpha remain unchanged.
/// expand_luminance replicates logical Y into RGB with alpha one, preserving
/// explicit R/RG and existing YA handling.
/// generate_mips generates missing mipmaps where supported; authored mips remain
/// without this flag.
///
/// raw (none) disables these optional operations; file decoding, normalization and
/// backend padding still apply. data generates mips only. data_rgb also expands
/// luminance. normal is an alias of data_rgb and uses ordinary mip generation for
/// now. color adds sRGB decoding to data_rgb and is the default. environment enables
/// sRGB decoding and luminance expansion without generating mips.
enum class TextureLoadFlags : uint32_t {
    /// No optional texture transformations.
    none = 0,
    /// Interpret supported 8-bit color inputs as sRGB regardless of metadata. HDR data and alpha are unchanged.
    srgb = 1 << 0,
    /// Expand logical luminance (Y) to RGB with alpha one. Explicit R/RG and existing YA handling are unchanged.
    expand_luminance = 1 << 1,
    /// Generate missing mipmaps where supported. Authored mip chains are retained without this flag.
    generate_mips = 1 << 2,
    /// No optional transformations. File decoding, normalization and backend padding still apply.
    raw = none,
    /// Mipmapped numeric data without sRGB decoding or luminance expansion.
    data = generate_mips,
    /// Mipmapped numeric data with luminance expanded to RGB.
    data_rgb = generate_mips | expand_luminance,
    /// Normal-map preset, equivalent to data_rgb. Uses ordinary mip generation for now.
    normal = data_rgb,
    /// Mipmapped color with sRGB decoding and luminance expansion.
    color = data_rgb | srgb,
    /// Environment color with sRGB decoding and luminance expansion, without generated mipmaps.
    environment = srgb | expand_luminance,
};
FALCOR_ENUM_CLASS_OPERATORS(TextureLoadFlags);
SGL_ENUM_FLAGS_INFO(
    TextureLoadFlags,
    {
        {TextureLoadFlags::none, "none"},
        {TextureLoadFlags::srgb, "srgb"},
        {TextureLoadFlags::expand_luminance, "expand_luminance"},
        {TextureLoadFlags::generate_mips, "generate_mips"},
        {TextureLoadFlags::raw, "raw"},
        {TextureLoadFlags::data, "data"},
        {TextureLoadFlags::data_rgb, "data_rgb"},
        {TextureLoadFlags::normal, "normal"},
        {TextureLoadFlags::color, "color"},
        {TextureLoadFlags::environment, "environment"},
    }
);
SGL_ENUM_REGISTER(TextureLoadFlags);

/// Translate texture-load intent consistently for file and embedded asset loaders.
inline sgl::TextureLoader::Options texture_loader_options(
    TextureLoadFlags flags,
    sgl::TextureUsage usage = sgl::TextureUsage::shader_resource,
    uint32_t max_mip_count = 0
)
{
    return {
        .srgb_mode = is_set(flags, TextureLoadFlags::srgb) ? sgl::SRGBMode::srgb : sgl::SRGBMode::linear,
        .y_handling = is_set(flags, TextureLoadFlags::expand_luminance) ? sgl::YHandling::expand_to_rgba
                                                                        : sgl::YHandling::preserve_as_r,
        .generate_mips = is_set(flags, TextureLoadFlags::generate_mips),
        .max_mip_count = max_mip_count,
        .usage = usage,
    };
}

} // namespace falcor
