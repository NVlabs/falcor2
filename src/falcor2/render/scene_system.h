// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/fwd.h"
#include "falcor2/render/scene_update.h"

#include "falcor2/core/object.h"

#include <sgl/device/fwd.h>

namespace falcor {

/// Base class for scene systems.
class SceneSystem : public Object {
    FALCOR_OBJECT(SceneSystem)
public:
    /// Constructor.
    /// @param scene The scene this system belongs to.
    SceneSystem(Scene* scene)
        : m_scene(scene)
    {
    }

    /// The scene this system belongs to.
    Scene* scene() const { return m_scene; }

    /// Called during Scene::update() to update the system.
    /// @param ctx Update context.
    /// @return Flags indicating what was updated.
    virtual SceneUpdateFlags update(SceneUpdateContext& ctx) = 0;

    /// Bind the system to the given shader cursor.
    /// @param cursor Shader cursor to bind to.
    virtual void bind_to_scene(const sgl::ShaderCursor& cursor) const = 0;

protected:
    /// Scene this system belongs to.
    Scene* m_scene{nullptr};

private:
    FALCOR_NON_COPYABLE_AND_MOVABLE(SceneSystem);
};

} // namespace falcor
