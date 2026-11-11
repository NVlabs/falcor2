// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "scene_object.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/component/camera.h"

namespace falcor {

// ----------------------------------------------------------------------------
// SceneObject
// ----------------------------------------------------------------------------

SceneObject::SceneObject() { }

SceneObject::~SceneObject() { }

void SceneObject::set_name(std::string_view name)
{
    FALCOR_CHECK(!is_builtin(), "Cannot rename a built-in object.");
    m_name = name;
}

bool SceneObject::is_builtin() const
{
    const Camera* free_camera = m_scene ? m_scene->free_camera() : nullptr;
    return free_camera && (this == free_camera || this == free_camera->entity());
}

void SceneObject::remove()
{
    FALCOR_CHECK(!is_builtin(), "Cannot remove a built-in object.");
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

void SceneObject::_set_source_asset_ref(std::optional<SourceAssetRef> source_asset_ref)
{
    m_source_asset_ref = source_asset_ref ? std::make_unique<SourceAssetRef>(std::move(*source_asset_ref)) : nullptr;
}

} // namespace falcor
