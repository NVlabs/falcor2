// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "entity.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/geometry.h"

namespace falcor {

Entity::~Entity() { }

void Entity::remove()
{
    for (Entity* child : m_children) {
        child->remove();
    }
    for (Component* component : m_components) {
        component->remove();
    }
    SceneObject::remove();
    m_scene->_mark_entity_removed(this);
}

void Entity::clear_invalid_references()
{
    erase_invalid_objects(m_children);
    erase_invalid_objects(m_components);
}

void Entity::on_add_to_scene()
{
    RenderTransformDesc desc = {};
    desc.world_from_object = world_from_object_matrix();
    m_render_transform = m_scene->_render_scene()->create_transform(desc);
}

void Entity::on_remove_from_scene()
{
    m_scene->_render_scene()->destroy_transform(m_render_transform);
    m_render_transform = {};
}

void Entity::set_parent(Entity* parent)
{
    if (parent == m_parent)
        return;

    if (m_parent) {
        auto it = std::find(m_parent->m_children.begin(), m_parent->m_children.end(), this);
        FALCOR_ASSERT(it != m_parent->m_children.end());
        m_parent->m_children.erase(it);
    }

    m_parent = parent;

    if (m_parent)
        m_parent->m_children.push_back(this);

    transform_changed();
}

void Entity::set_transform(const Transform& transform)
{
    m_transform = transform;
    transform_changed();
}

void Entity::set_world_transform(const Transform& world_transform)
{
    if (m_parent) {
        float4x4 local_matrix = mul(sgl::math::inverse(m_parent->world_from_object_matrix()), world_transform.matrix());
        set_transform(Transform(local_matrix));
    } else {
        set_transform(world_transform);
    }
}

float4x4 Entity::world_from_object_matrix() const
{
    if (m_world_from_object_matrix_dirty) {
        m_world_from_object_matrix
            = m_parent ? mul(m_parent->world_from_object_matrix(), m_transform.matrix()) : m_transform.matrix();
        m_world_from_object_matrix_dirty = false;
    }
    return m_world_from_object_matrix;
}

AABB Entity::world_aabb() const
{
    float4x4 world_matrix = world_from_object_matrix();
    AABB result;
    for (const Component* component : m_components) {
        const GeometryInstance* geo_inst = component->as<const GeometryInstance>();
        if (!geo_inst)
            continue;
        const Geometry* geometry = geo_inst->geometry();
        if (!geometry || !geometry->local_aabb().is_valid())
            continue;
        AABB world = geometry->local_aabb().transform(world_matrix);
        if (!result.is_valid())
            result = world;
        else
            result.expand(world);
    }
    return result;
}

Component* Entity::create_component(std::string_view type, std::optional<Properties> props)
{
    Component* component = m_scene->_create_object<Component>(type, props);
    component->m_entity = this;
    m_components.push_back(component);
    return component;
}

Component* Entity::create_component(const std::type_info& type, std::optional<Properties> props)
{
    Component* component = m_scene->_create_object<Component>(type, props);
    component->m_entity = this;
    m_components.push_back(component);
    return component;
}

void Entity::update_transform()
{
    m_scene->_render_scene()->update_transform(m_render_transform, world_from_object_matrix());
}

void Entity::mark_dirty(DirtyFlags flags)
{
    m_scene->_mark_entity_dirty(this, flags);
}

void Entity::transform_changed()
{
    mark_dirty(DirtyFlags::transform);
    m_world_from_object_matrix_dirty = true;
    for (Component* component : m_components) {
        component->mark_dirty(Component::DirtyFlags::entity_transform);
        component->on_entity_transform_changed();
    }
    for (Entity* child : m_children)
        child->transform_changed();
}

FALCOR_STATIC_ONCE(reflection::register_type<Entity>());

} // namespace falcor
