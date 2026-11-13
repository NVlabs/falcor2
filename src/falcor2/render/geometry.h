// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_object.h"
#include "falcor2/render/render_scene.h"
#include "falcor2/utils/aabb.h"

namespace falcor {

class Geometry;

/// Bounds measured from mesh vertices in one coordinate system.
/// Empty bounds have an invalid AABB and a zero radius. A single point has a valid AABB.
struct FALCOR_API MeshBounds {
    AABB aabb;
    /// Maximum distance from the coordinate system's origin to a mesh vertex.
    float radius{0.f};

    /// Include a finite point. Throws if its distance cannot be represented as a float.
    void expand(const float3& point);
    /// Include bounds measured in the same coordinate system. Empty bounds are ignored.
    void expand(const MeshBounds& other);
};

namespace detail {
void validate_bounds_transform(const float4x4& transform);
}

/// Base class for scene geometry.
class FALCOR_API Geometry : public SceneObject {
    FALCOR_SCENE_OBJECT(Geometry, SceneObject)
public:
    /// Dirty flags for Geometry.
    enum class DirtyFlags : uint32_t {
        none = 0,
        render_state = (1u << 0),
        /// Vertex data changed without replacing render state or topology.
        deformation = (1u << 1),
        // Shared SceneObject::DirtyFlags
        resources = (1u << 29),
        added = (1u << 30),
        removed = (1u << 31),
    };

    /// Constructor.
    Geometry();

    /// Destructor.
    virtual ~Geometry();

    // SceneObject interface

    virtual SceneObjectKind kind() const override { return SceneObjectKind::geometry; }

    virtual void remove() override;

    // Geometry interface

    /// Called during Scene::update() to update the render state of the component.
    virtual void update_render_state() { }

    /// Local-space axis-aligned bounding box of the geometry.
    const AABB& local_aabb() const { return m_local_aabb; }

    /// Calculate exact mesh bounds after applying a finite affine transform to local vertices.
    /// Empty geometry returns empty bounds. Geometry types without supported CPU vertex data throw.
    virtual MeshBounds calculate_bounds(const float4x4& transform = float4x4::identity()) const;

    const RenderGeometryGroupHandle& _render_geometry_group() const { return m_render_geometry_group; }

protected:
    void mark_dirty(DirtyFlags flags);

    /// Local-space axis-aligned bounding box.
    AABB m_local_aabb;

    /// Render geometry group representing this geometry.
    RenderGeometryGroupHandle m_render_geometry_group;
};

FALCOR_ENUM_CLASS_OPERATORS(Geometry::DirtyFlags);

#define FALCOR_SCENE_REGISTER_GEOMETRY(name) FALCOR_SCENE_REGISTER_CLASS(name, Geometry)

} // namespace falcor
