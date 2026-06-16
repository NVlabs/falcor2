// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_object.h"

#include "falcor2/core/enum.h"

namespace falcor {

/// Base class for scene components.
class FALCOR_API Component : public SceneObject {
    FALCOR_SCENE_OBJECT(Component, SceneObject)
public:
    /// Dirty flags for Component.
    enum class DirtyFlags : uint32_t {
        none = 0,
        entity_transform = (1u << 0),
        render_state = (1u << 1),
        animation_state = (1u << 2),
        // Shared SceneObject::DirtyFlags
        resources = (1u << 29),
        added = (1u << 30),
        removed = (1u << 31),
    };

    /// Constructor.
    Component();

    /// Destructor.
    virtual ~Component();

    // SceneObject interface

    virtual SceneObjectKind kind() const override { return SceneObjectKind::component; }

    virtual void remove() override;

    // Component interface

    /// The entity this component belongs to.
    Entity* entity() const { return m_entity; }

    /// Called immediately when entity transform changed.
    virtual void on_entity_transform_changed() { }

    /// Called during Scene::update() to update the render state of the component.
    virtual void update_render_state() { }

protected:
    void mark_dirty(DirtyFlags flags);

    /// The entity this component belongs to.
    Entity* m_entity{nullptr};

private:
    friend class Entity;
};

FALCOR_ENUM_CLASS_OPERATORS(Component::DirtyFlags);

#define FALCOR_SCENE_REGISTER_COMPONENT(name) FALCOR_SCENE_REGISTER_CLASS(name, Component)

} // namespace falcor
