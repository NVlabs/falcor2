// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/ui/icon_library.h"

#include "falcor2/core/error.h"

#include <cmrc/cmrc.hpp>

#include <sgl/core/memory_stream.h>
#include <sgl/device/resource.h>
#include <sgl/utils/texture_loader.h>

#include <array>
#include <cstddef>
#include <utility>

CMRC_DECLARE(falcor2_ui_data);

namespace falcor::ui {
namespace {

struct IconMetadata {
    float2 uv_min;
    float2 uv_max;
    float preferred_size;
};

#define FALCOR_UI_ICON_ATLAS(width, height)                                                                            \
    constexpr uint32_t ICON_ATLAS_WIDTH = width;                                                                       \
    constexpr uint32_t ICON_ATLAS_HEIGHT = height;
#define FALCOR_UI_ICON(name, x, y, width, height, preferred_size)
#include "falcor2/ui/icon_data.inc"
#undef FALCOR_UI_ICON
#undef FALCOR_UI_ICON_ATLAS

constexpr float HALF_TEXEL = 0.5f;
constexpr std::array<IconMetadata, std::size_t(Icon::_count)> ICON_METADATA{
#define FALCOR_UI_ICON_ATLAS(width, height)
#define FALCOR_UI_ICON(name, x, y, width, height, preferred_size)                                                      \
    IconMetadata{                                                                                                      \
        float2((float(x) + HALF_TEXEL) / float(ICON_ATLAS_WIDTH), (float(y) + HALF_TEXEL) / float(ICON_ATLAS_HEIGHT)), \
        float2(                                                                                                        \
            (float(x) + float(width) - HALF_TEXEL) / float(ICON_ATLAS_WIDTH),                                          \
            (float(y) + float(height) - HALF_TEXEL) / float(ICON_ATLAS_HEIGHT)                                         \
        ),                                                                                                             \
        preferred_size                                                                                                 \
    },
#include "falcor2/ui/icon_data.inc"
#undef FALCOR_UI_ICON
#undef FALCOR_UI_ICON_ATLAS
};

} // namespace

IconLibrary::IconLibrary(ref<sgl::Device> device)
    : m_device(std::move(device))
{
    FALCOR_CHECK(m_device, "Cannot create an icon library without a device.");

    const auto filesystem = cmrc::falcor2_ui_data::get_filesystem();
    const auto resource = filesystem.open("ui/assets/icon_atlas.png");
    sgl::MemoryStream stream(resource.begin(), resource.size());
    sgl::TextureLoader loader(m_device);
    sgl::TextureLoader::Options options;
    options.load_as_srgb = false;
    m_texture = loader.load_texture(&stream, options);
    FALCOR_CHECK(
        m_texture && m_texture->width() == ICON_ATLAS_WIDTH && m_texture->height() == ICON_ATLAS_HEIGHT,
        "Failed to load the embedded icon atlas."
    );
}

IconImage IconLibrary::icon(Icon icon) const
{
    const std::size_t index = std::size_t(icon);
    FALCOR_CHECK(index < ICON_METADATA.size(), "Invalid icon {}.", uint32_t(icon));
    const IconMetadata& metadata = ICON_METADATA[index];
    return {
        .texture = m_texture.get(),
        .uv_min = metadata.uv_min,
        .uv_max = metadata.uv_max,
        .preferred_size = metadata.preferred_size,
    };
}

} // namespace falcor::ui
