// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry.h"

#include "falcor2/render/scene.h"

namespace falcor {

Geometry::Geometry() { }

Geometry::~Geometry() { }

void Geometry::remove()
{
    SceneObject::remove();
    m_scene->_mark_geometry_removed(this);
}

void Geometry::mark_dirty(DirtyFlags flags)
{
    m_scene->_mark_geometry_dirty(this, flags);
}

} // namespace falcor
