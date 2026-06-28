// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry_instance.h"

#include "falcor2/render/scene.h"

namespace falcor {

GeometryInstance::~GeometryInstance() { }

void GeometryInstance::set_geometry(Geometry* geometry)
{
    if (geometry != m_geometry) {
        m_geometry = geometry;
        mark_dirty(Component::DirtyFlags::render_state);
    }
}

void GeometryInstance::set_materials(const std::vector<Material*>& materials)
{
    if (materials != m_materials) {
        m_materials = materials;
        mark_dirty(Component::DirtyFlags::render_state);
    }
}

shared::GeometryInstanceID GeometryInstance::geometry_instance_id() const
{
    if (m_render_geometry_instance)
        return m_scene->_render_scene()->get_geometry_instance(m_render_geometry_instance).geometry_instance_id;
    return shared::GeometryInstanceID::invalid;
}

uint32_t GeometryInstance::geometry_instance_count() const
{
    if (m_render_geometry_instance)
        return m_scene->_render_scene()->get_geometry_instance(m_render_geometry_instance).geometry_instance_count;
    return 0;
}

void GeometryInstance::clear_invalid_references()
{
    if (m_geometry && !m_geometry->is_valid())
        set_geometry(nullptr);

    // Remove invalid materials.
    {
        bool changed = false;
        for (Material*& material : m_materials) {
            if (material && !material->is_valid()) {
                material = nullptr;
                changed = true;
            }
        }
        if (changed)
            mark_dirty(DirtyFlags::render_state);
    }
}

void GeometryInstance::on_add_to_scene()
{
    mark_dirty(Component::DirtyFlags::render_state);
}

void GeometryInstance::on_remove_from_scene()
{
    RenderScene* render_scene = m_scene->_render_scene();

    if (m_render_geometry_instance) {
        render_scene->destroy_geometry_instance(m_render_geometry_instance);
        m_render_geometry_instance = {};
    }
}

void GeometryInstance::update_render_state()
{
    RenderScene* render_scene = m_scene->_render_scene();

    if (m_render_geometry_instance) {
        render_scene->destroy_geometry_instance(m_render_geometry_instance);
        m_render_geometry_instance = {};
    }

    if (m_geometry) {
        RenderGeometryGroupHandle geometry_group = m_geometry->_render_geometry_group();
        if (geometry_group) {
            RenderGeometryInstanceDesc desc = {};
            desc.geometry_group = geometry_group;
            desc.transform = m_entity->_render_transform();
            std::vector<shared::MaterialID> material_ids(m_materials.size());
            for (size_t i = 0; i < m_materials.size(); ++i) {
                Material* material = m_materials[i];
                material_ids[i] = (material && material->is_valid()) ? material->material_id() : shared::MaterialID{0};
            }
            desc.material_ids = material_ids;
            m_render_geometry_instance = render_scene->create_geometry_instance(desc);
        }
    }
}

FALCOR_SCENE_REGISTER_COMPONENT(GeometryInstance);

} // namespace falcor
