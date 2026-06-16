// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/types.h"

#include "falcor2/render/fwd.h"

#include <string>

namespace falcor::ui {

struct ViewportOverlayDesc {
    float2 screen_pos{0.f, 0.f};
    float2 screen_size{0.f, 0.f};
    float fps{0.f};
};

/// Draw a viewport overlay containing an FPS counter.
FALCOR_API void draw_viewport_overlay(const ViewportOverlayDesc& desc);

FALCOR_API bool toggle_button(const char* label, bool active);

/// ImGui widget for editing a Transform (translation, rotation in degrees, scale).
/// Returns true if the transform was modified.
FALCOR_API bool transform_editor(const char* label, Transform& transform);

} // namespace falcor::ui
