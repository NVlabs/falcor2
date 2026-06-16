// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"
#include "falcor2/core/types.h"

#include <sgl/core/input.h>

#include <functional>

namespace falcor {
class Scene;
class Camera;
} // namespace falcor

namespace falcor::ui {

class CameraController;
class SceneEditor;
class ScenePicker;
class SelectionOverlay;

/// Coordinates input routing between the scene editor, camera controller, picker,
/// and selection overlay.
///
/// Centralizes the input routing order, viewport-local coordinate remapping,
/// gizmo suppression, camera capture side effects, and selection picking that
/// would otherwise be scattered across the application.
///
/// The controller does not own any of its services; they are injected by the
/// application and can be null (in which case the corresponding feature is
/// disabled).
class FALCOR_API SceneInteractionController : public Object {
    FALCOR_OBJECT(SceneInteractionController)
public:
    /// Callback invoked when the controller requests accumulation reset.
    using ResetCallback = std::function<void()>;

    SceneInteractionController(
        ref<SceneEditor> scene_editor,
        ref<ScenePicker> scene_picker,
        ref<SelectionOverlay> selection_overlay,
        ref<CameraController> camera_controller
    );

    /// Scene editor used for gizmos and selection.
    SceneEditor* scene_editor() const { return m_scene_editor; }

    /// Scene picker used for viewport picking.
    ScenePicker* scene_picker() const { return m_scene_picker; }

    /// Selection overlay used for selection highlighting.
    SelectionOverlay* selection_overlay() const { return m_selection_overlay; }

    /// Camera controller used for navigation.
    CameraController* camera_controller() const { return m_camera_controller; }

    /// Scene for picking operations.
    Scene* scene() const { return m_scene; }
    void set_scene(ref<Scene> scene);

    /// Callback invoked when interaction requires accumulation reset (e.g. selection change, focus on selection).
    ResetCallback reset_callback() const { return m_reset_callback; }
    void set_reset_callback(ResetCallback callback);

    /// Process a keyboard event. Routes through capture state, UI, and camera controller.
    /// @return True if the event was consumed.
    bool handle_keyboard_event(const sgl::KeyboardEvent& event);

    /// Process a mouse event. Routes through capture state, UI, picker, and camera controller.
    /// @return True if the event was consumed.
    bool handle_mouse_event(const sgl::MouseEvent& event);

    /// Update selection overlay if the editor's selection changed.
    void update_selection_overlay();

    /// Focus the camera on the currently selected entity.
    /// @param camera The camera to update, or nullptr if no camera is active.
    /// @return True if the camera was moved.
    bool focus_on_selection(Camera* camera);

private:
    void request_reset();

    ref<SceneEditor> m_scene_editor;
    ref<ScenePicker> m_scene_picker;
    ref<SelectionOverlay> m_selection_overlay;
    ref<CameraController> m_camera_controller;

    ref<Scene> m_scene;

    uint64_t m_last_selection_version{0};

    ResetCallback m_reset_callback;
};

} // namespace falcor::ui
