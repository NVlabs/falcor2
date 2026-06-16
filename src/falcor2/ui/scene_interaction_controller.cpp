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

bool SceneInteractionController::handle_keyboard_event(const sgl::KeyboardEvent& event)
{
    if (!m_camera_controller)
        return false;

    // When captured, route all keyboard events to the camera controller.
    // If the controller releases capture during handling, fall through to let
    // the rest of the input chain process the event.
    if (m_camera_controller->is_captured()) {
        m_camera_controller->handle_keyboard_event(event);
        return m_camera_controller->is_captured();
    }

    // Non-captured keyboard events are forwarded to the camera controller
    // (for modifier tracking etc.) but are not consumed.
    m_camera_controller->handle_keyboard_event(event);
    return false;
}

bool SceneInteractionController::handle_mouse_event(const sgl::MouseEvent& event)
{
    if (!m_camera_controller)
        return false;

    // When captured, route all mouse events to the camera controller.
    if (m_camera_controller->is_captured()) {
        m_camera_controller->handle_mouse_event(event);
        return m_camera_controller->is_captured();
    }

    m_camera_controller->handle_mouse_event(event);

    // Handle picking on left-click in the viewport, but only if the camera
    // controller didn't just enter an active mode (e.g. Alt+LMB orbit).
    if (!m_camera_controller->is_captured() && m_scene_picker && m_scene_editor && m_scene && event.is_button_down()
        && event.button == sgl::MouseButton::left) {
        uint2 local_pos;
        if (m_scene_editor->can_pick_at(event.pos, local_pos)) {
            Entity* entity = m_scene_picker->pick_entity(m_scene, local_pos);
            m_scene_editor->set_selected_object(entity);
        }
    }

    return false;
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

} // namespace falcor::ui
