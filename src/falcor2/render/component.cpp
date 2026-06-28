// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "component.h"

#include "falcor2/render/scene.h"

namespace falcor {

Component::Component() { }

Component::~Component() { }

void Component::remove()
{
    SceneObject::remove();
    m_scene->_mark_component_removed(this);
}

void Component::mark_dirty(DirtyFlags flags)
{
    m_scene->_mark_component_dirty(this, flags);
}

} // namespace falcor
