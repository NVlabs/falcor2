// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"
#include "falcor2/core/types.h"

#include <sgl/device/fwd.h>

#include <cstdint>

namespace falcor::ui {

/// Icons available from the shared atlas.
enum class Icon : uint32_t {
#define FALCOR_UI_ICON_ATLAS(width, height)
#define FALCOR_UI_ICON(name, x, y, width, height, preferred_size) name,
#include "falcor2/ui/icon_data.inc"
#undef FALCOR_UI_ICON
#undef FALCOR_UI_ICON_ATLAS
    _count,
};

/// Texture and layout information needed to draw one icon.
struct IconImage {
    sgl::Texture* texture{nullptr};
    float2 uv_min{0.f};
    float2 uv_max{1.f};
    float preferred_size{0.f};
};

/// Per-device texture resources and metadata shared by icon consumers.
class FALCOR_API IconLibrary : public Object {
    FALCOR_OBJECT(IconLibrary)
public:
    explicit IconLibrary(ref<sgl::Device> device);

    sgl::Device* device() const { return m_device.get(); }
    sgl::Texture* texture() const { return m_texture.get(); }

    IconImage icon(Icon icon) const;

private:
    ref<sgl::Device> m_device;
    ref<sgl::Texture> m_texture;
};

} // namespace falcor::ui
