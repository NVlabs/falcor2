// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_object.h"
#include "falcor2/render/render_scene.h"
#include "falcor2/utils/aabb.h"

namespace falcor {

class Geometry;

/// Base class for scene geometry.
class FALCOR_API Geometry : public SceneObject {
    FALCOR_SCENE_OBJECT(Geometry, SceneObject)
public:
    /// Dirty flags for Geometry.
    enum class DirtyFlags : uint32_t {
        none = 0,
        render_state = (1u << 0),
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

    const RenderGeometryGroupHandle& _render_geometry_group() const { return m_render_geometry_group; }

protected:
    void mark_dirty(DirtyFlags flags);

    /// Local-space axis-aligned bounding box.
    AABB m_local_aabb;

    /// List of geometry instances referencing this geometry.
    std::vector<GeometryInstance*> m_geometry_instances;

    /// Render geometry group representing this geometry.
    RenderGeometryGroupHandle m_render_geometry_group;
};

FALCOR_ENUM_CLASS_OPERATORS(Geometry::DirtyFlags);

#define FALCOR_SCENE_REGISTER_GEOMETRY(name) FALCOR_SCENE_REGISTER_CLASS(name, Geometry)

} // namespace falcor
