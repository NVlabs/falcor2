// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_object.h"
#include "falcor2/render/transform.h"
#include "falcor2/render/render_scene.h"
#include "falcor2/utils/aabb.h"

#include "falcor2/core/properties.h"

namespace falcor {

/// Scene entity.
/// An entity represents an object in the scene graph with a transform and a set of components.
class FALCOR_API Entity : public SceneObject {
    FALCOR_SCENE_OBJECT(Entity, SceneObject)
public:
    /// Dirty flags for Entity.
    enum class DirtyFlags : uint32_t {
        none = 0,
        transform = (1u << 0),
        // Shared SceneObject::DirtyFlags
        resources = (1u << 29),
        added = (1u << 30),
        removed = (1u << 31),
    };

    /// Destructor.
    virtual ~Entity();

    // SceneObject interface

    virtual SceneObjectKind kind() const override { return SceneObjectKind::entity; }

    virtual void remove() override;

    virtual void clear_invalid_references() override;

    virtual void on_add_to_scene() override;
    virtual void on_remove_from_scene() override;

    // Entity interface

    /// Parent entity or nullptr if this is a root entity.
    Entity* parent() const { return m_parent; }
    void set_parent(Entity* parent);

    /// List of child entities.
    const std::vector<Entity*>& children() const { return m_children; }

    /// List of components belonging to this entity.
    const std::vector<Component*>& components() const { return m_components; }

    /// Transform of the entity (local space, relative to parent).
    const Transform& transform() const { return m_transform; }
    void set_transform(const Transform& transform);

    /// Set the world-space transform of this entity.
    /// If the entity has a parent, the local transform is computed by
    /// multiplying with the inverse of the parent's world matrix.
    void set_world_transform(const Transform& world_transform);

    /// World-from-object transformation matrix.
    float4x4 world_from_object_matrix() const;

    /// Compute the world-space axis-aligned bounding box of this entity.
    /// Returns an invalid AABB if the entity has no geometry.
    AABB world_aabb() const;

    /// Create a component instance.
    /// @tparam T Type of component to create.
    /// @param props Properties to initialize the component (optional).
    /// @return The created component.
    template<typename T>
        requires std::is_base_of_v<Component, T>
    T* create_component(std::optional<Properties> props = {});

    /// Create a component instance.
    /// @param type Type name of the component to create.
    /// @param props Properties to initialize the component (optional).
    /// @return The created component.
    Component* create_component(std::string_view type, std::optional<Properties> props = {});

    /// Create a component instance.
    /// @param type Type info of the component to create.
    /// @param props Properties to initialize the component (optional).
    /// @return The created component.
    Component* create_component(const std::type_info& type, std::optional<Properties> props = {});

    /// Called during Scene::update() to update the transform.
    void update_transform();

    RenderTransformHandle _render_transform() const { return m_render_transform; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        FALCOR_UNUSED(r);
    }

protected:
    void mark_dirty(DirtyFlags flags);

private:
    /// Called immediately when the transform of this entity changed, or the entity was reparented.
    /// This marks the transform as dirty, and notifies components and child entities about the change.
    void transform_changed();

    Entity* m_parent{nullptr};
    std::vector<Entity*> m_children;
    std::vector<Component*> m_components;
    Transform m_transform;
    mutable float4x4 m_world_from_object_matrix;
    mutable bool m_world_from_object_matrix_dirty{true};

    RenderTransformHandle m_render_transform;
};

FALCOR_ENUM_CLASS_OPERATORS(Entity::DirtyFlags);

} // namespace falcor
