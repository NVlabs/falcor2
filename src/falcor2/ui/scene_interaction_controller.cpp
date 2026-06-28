// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "scene_interaction_controller.h"

#include "falcor2/ui/camera_controller.h"
#include "falcor2/ui/scene_editor.h"
#include "falcor2/ui/scene_picker.h"
#include "falcor2/ui/selection_overlay.h"

#include "falcor2/render/entity.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/utils/aabb.h"

#include <sgl/math/vector_math.h>

namespace falcor::ui {

SceneInteractionController::SceneInteractionController(
    ref<SceneEditor> scene_editor,
    ref<ScenePicker> scene_picker,
    ref<SelectionOverlay> selection_overlay,
    ref<CameraController> camera_controller
)
    : m_scene_editor(std::move(scene_editor))
    , m_scene_picker(std::move(scene_picker))
    , m_selection_overlay(std::move(selection_overlay))
    , m_camera_controller(std::move(camera_controller))
{
}

void SceneInteractionController::set_scene(ref<Scene> scene)
{
    if (scene != m_scene) {
        m_scene = std::move(scene);
        // Prevent dangling selection overlay reference to entities from previous scene.
        if (m_selection_overlay)
            m_selection_overlay->clear_selection();
    }
}

void SceneInteractionController::set_reset_callback(ResetCallback callback)
{
    m_reset_callback = std::move(callback);
}

void SceneInteractionController::set_pointer_capture_callback(PointerCaptureCallback callback)
{
    m_pointer_capture_callback = std::move(callback);
}

SceneInteractionController::PointerOwner SceneInteractionController::pointer_owner()
{
    reconcile_pointer_owner();
    return m_pointer_owner;
}

bool SceneInteractionController::has_pointer_capture()
{
    reconcile_pointer_owner();
    return m_pointer_owner != PointerOwner::none;
}

void SceneInteractionController::cancel_pointer_owner()
{
    if (m_camera_controller)
        m_camera_controller->cancel_interaction();
    set_pointer_owner(PointerOwner::none);
}

bool SceneInteractionController::handle_keyboard_event(const sgl::KeyboardEvent& event)
{
    if (!m_camera_controller)
        return false;

    // The camera always sees keyboard events so movement/modifier state stays coherent,
    // but it only consumes keys it actually owns in its current mode.
    return m_camera_controller->handle_keyboard_event(event);
}

bool SceneInteractionController::handle_mouse_event(const sgl::MouseEvent& event)
{
    if (!m_camera_controller)
        return false;

    reconcile_pointer_owner();

    if (m_pointer_owner == PointerOwner::camera) {
        m_camera_controller->handle_mouse_event(event);
        if (!m_camera_controller->is_interacting())
            set_pointer_owner(PointerOwner::none);
        return true;
    }

    bool route_allowed = can_route_viewport_event(event);
    bool camera_handled = false;
    if (route_allowed) {
        camera_handled = m_camera_controller->handle_mouse_event(event);
        if (m_camera_controller->is_interacting())
            set_pointer_owner(PointerOwner::camera);
    }

    // Handle picking on left-click in the viewport, but only if the camera
    // controller didn't just enter an active mode (e.g. Alt+LMB orbit).
    if (route_allowed && !camera_handled && m_scene_picker && m_scene_editor && m_scene && event.is_button_down()
        && event.button == sgl::MouseButton::left) {
        uint2 local_pos;
        if (m_scene_editor->can_pick_at(event.pos, local_pos)) {
            Entity* entity = m_scene_picker->pick_entity(m_scene, local_pos);
            m_scene_editor->set_selected_object(entity);
            camera_handled = true;
        }
    }

    return camera_handled;
}

void SceneInteractionController::update_selection_overlay()
{
    if (!m_scene_editor || !m_selection_overlay)
        return;

    uint64_t version = m_scene_editor->selection_version();
    if (version != m_last_selection_version) {
        m_last_selection_version = version;
        Entity* entity = m_scene_editor->selected_object() ? m_scene_editor->selected_object()->as<Entity>() : nullptr;
        m_selection_overlay->set_selected_entity(entity);
    }
}

bool SceneInteractionController::focus_on_selection(Camera* camera)
{
    if (!m_scene_editor || !m_camera_controller)
        return false;

    if (!m_scene_editor->selected_object())
        return false;

    Entity* entity = m_scene_editor->selected_object()->as<Entity>();
    if (!entity)
        return false;

    AABB aabb = entity->world_aabb();
    if (!aabb.is_valid())
        return false;

    float distance = sgl::math::length(aabb.size()) * 1.5f;
    m_camera_controller->focus(aabb.center(), distance);
    if (camera)
        camera->entity()->set_world_transform(m_camera_controller->transform());
    request_reset();
    return true;
}

void SceneInteractionController::request_reset()
{
    if (m_reset_callback)
        m_reset_callback();
}

void SceneInteractionController::set_pointer_owner(PointerOwner owner)
{
    if (m_pointer_owner == owner)
        return;

    m_pointer_owner = owner;
    if (m_pointer_capture_callback)
        m_pointer_capture_callback(m_pointer_owner != PointerOwner::none);
}

void SceneInteractionController::reconcile_pointer_owner()
{
    if (m_pointer_owner == PointerOwner::camera && (!m_camera_controller || !m_camera_controller->is_interacting()))
        set_pointer_owner(PointerOwner::none);
}

bool SceneInteractionController::can_route_viewport_event(const sgl::MouseEvent& event) const
{
    if (!m_scene_editor || !m_scene_editor->visible())
        return true;

    if (!m_scene_editor->is_viewport_interactive())
        return false;

    bool gizmo_active = m_scene_editor->is_gizmo_active();
    if (!gizmo_active)
        return true;

    // ImGuizmo owns left-button interaction and active drags, but RMB/MMB
    // navigation may still start while the cursor merely hovers the gizmo.
    return !(event.is_move() || event.button == sgl::MouseButton::left);
}

} // namespace falcor::ui
