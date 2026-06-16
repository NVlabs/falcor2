// SPDX-License-Identifier: Apache-2.0

#include "scene_object.h"

#include "falcor2/render/scene.h"

namespace falcor {

// ----------------------------------------------------------------------------
// SceneObject
// ----------------------------------------------------------------------------

SceneObject::SceneObject() { }

SceneObject::~SceneObject() { }

void SceneObject::set_name(std::string_view name)
{
    m_name = name;
}

void SceneObject::remove()
{
    m_removed = true;
}

void SceneObject::on_create(std::optional<const Properties> props)
{
    if (props)
        set_properties(*props);
}

void SceneObject::on_destroy() { }

void SceneObject::_create(std::optional<const Properties> props)
{
    on_create(props);
}

void SceneObject::_destroy()
{
    FALCOR_ASSERT(m_removed);
    on_destroy();
    m_scene = nullptr;
    m_collection_index = INVALID_SCENE_OBJECT_COLLECTION_INDEX;
}

void SceneObject::_set_default_name()
{
    if (m_name.empty())
        m_name = fmt::format("{}.{}", sgl::enum_to_string(kind()), m_collection_index);
}

} // namespace falcor
