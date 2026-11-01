// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/render/fwd.h"
#include "falcor2/ui/editor_command.h"

#include <span>
#include <string_view>

namespace falcor::ui {

struct FALCOR_API SceneObjectTemplate {
    std::string_view key;
    std::string_view label;
    /// Slash-separated menu category; empty places the command at the root.
    std::string_view category;
    /// Create a scene-owned object; naming and editor selection are handled by the caller.
    SceneObject* (*create)(Scene&);

    CommandId command_id() const;
};

/// Return the built-in templates in menu order. The view and its entries have static lifetime.
FALCOR_API std::span<const SceneObjectTemplate> scene_object_templates();

} // namespace falcor::ui
