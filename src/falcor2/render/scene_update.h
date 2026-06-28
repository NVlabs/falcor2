// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/core/enum.h"

#include <sgl/device/fwd.h>

namespace falcor {

/// Context passed to scene update functions.
class FALCOR_API SceneUpdateContext {
public:
    SceneUpdateContext(sgl::Device* device);

    /// The device used for this update.
    sgl::Device* device() const { return m_device; }

    /// Command encoder to record commands into.
    sgl::CommandEncoder* command_encoder();

    /// Submit the recorded commands to the device.
    void submit();

private:
    sgl::Device* m_device;
    ref<sgl::CommandEncoder> m_command_encoder;

    SGL_NON_COPYABLE_AND_MOVABLE(SceneUpdateContext);
};

/// Flags indicating what was updated during a scene update.
enum class SceneUpdateFlags {
    none = 0,
    geometry = (1 << 0),
    materials = (1 << 1),
    lights = (1 << 2),
    transforms = (1 << 3),
    render_state = (1 << 4),
    requirements = (1 << 5),
    animation = (1 << 6),
};
FALCOR_ENUM_CLASS_OPERATORS(SceneUpdateFlags);
SGL_ENUM_FLAGS_INFO(
    SceneUpdateFlags,
    {
        {SceneUpdateFlags::none, "none"},
        {SceneUpdateFlags::geometry, "geometry"},
        {SceneUpdateFlags::materials, "materials"},
        {SceneUpdateFlags::lights, "lights"},
        {SceneUpdateFlags::transforms, "transforms"},
        {SceneUpdateFlags::render_state, "render_state"},
        {SceneUpdateFlags::requirements, "requirements"},
        {SceneUpdateFlags::animation, "animation"},
    }
);
SGL_ENUM_REGISTER(SceneUpdateFlags);

} // namespace falcor
