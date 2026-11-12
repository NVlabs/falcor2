// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry.h"

#include "falcor2/render/scene.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace falcor {

void MeshBounds::expand(const float3& point)
{
    FALCOR_CHECK(sgl::math::all(sgl::math::isfinite(point)), "Mesh bounds require finite vertex positions.");
    double point_radius = std::hypot(double(point.x), double(point.y), double(point.z));
    FALCOR_CHECK(
        point_radius <= std::numeric_limits<float>::max(),
        "Mesh bounds radius must be representable as a finite float."
    );
    aabb.expand(point);
    radius = std::max(radius, float(point_radius));
}

void MeshBounds::expand(const MeshBounds& other)
{
    if (!other.aabb.is_valid())
        return;
    aabb.expand(other.aabb);
    radius = std::max(radius, other.radius);
}

void detail::validate_bounds_transform(const float4x4& transform)
{
    for (int row = 0; row < 4; ++row)
        FALCOR_CHECK(sgl::math::all(sgl::math::isfinite(transform[row])), "Bounds transform must be finite.");
    FALCOR_CHECK(
        transform[3][0] == 0.f && transform[3][1] == 0.f && transform[3][2] == 0.f && transform[3][3] == 1.f,
        "Bounds transform must be affine."
    );
}

Geometry::Geometry() { }

Geometry::~Geometry() { }

void Geometry::remove()
{
    SceneObject::remove();
    m_scene->_mark_geometry_removed(this);
}

MeshBounds Geometry::calculate_bounds(const float4x4& transform) const
{
    detail::validate_bounds_transform(transform);
    FALCOR_THROW("Exact mesh bounds calculation is not supported for geometry '{}'.", name());
}

void Geometry::mark_dirty(DirtyFlags flags)
{
    m_scene->_mark_geometry_dirty(this, flags);
}

} // namespace falcor
