// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/ui/scene_interaction_controller.h"
#include "falcor2/ui/camera_controller.h"
#include "falcor2/ui/scene_editor.h"
#include "falcor2/ui/scene_picker.h"
#include "falcor2/ui/selection_overlay.h"
#include "falcor2/render/scene.h"
#include "falcor2/render/component/camera.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(ui_scene_interaction_controller)
{
    using namespace falcor;

    nb::module_ ui = nb::module_::import_("falcor2.ui");

    nb::class_<ui::SceneInteractionController, Object> scene_interaction_controller(
        ui,
        "SceneInteractionController",
        D(ui, SceneInteractionController)
    );
    scene_interaction_controller
        .def(
            nb::init<
                ref<ui::SceneEditor>,
                ref<ui::ScenePicker>,
                ref<ui::SelectionOverlay>,
                ref<ui::CameraController>>(),
            "scene_editor"_a.none(),
            "scene_picker"_a.none(),
            "selection_overlay"_a.none(),
            "camera_controller"_a.none(),
            D(ui, SceneInteractionController, SceneInteractionController)
        )
        .def_prop_ro(
            "scene_editor",
            &ui::SceneInteractionController::scene_editor,
            D(ui, SceneInteractionController, scene_editor)
        )
        .def_prop_ro(
            "scene_picker",
            &ui::SceneInteractionController::scene_picker,
            D(ui, SceneInteractionController, scene_picker)
        )
        .def_prop_ro(
            "selection_overlay",
            &ui::SceneInteractionController::selection_overlay,
            D(ui, SceneInteractionController, selection_overlay)
        )
        .def_prop_ro(
            "camera_controller",
            &ui::SceneInteractionController::camera_controller,
            D(ui, SceneInteractionController, camera_controller)
        )
        .def_prop_rw(
            "scene",
            &ui::SceneInteractionController::scene,
            &ui::SceneInteractionController::set_scene,
            nb::arg().none(),
            D(ui, SceneInteractionController, scene)
        )
        .def_prop_rw(
            "reset_callback",
            &ui::SceneInteractionController::reset_callback,
            &ui::SceneInteractionController::set_reset_callback,
            nb::arg().none(),
            D(ui, SceneInteractionController, reset_callback)
        )
        .def_prop_ro(
            "pointer_owner",
            &ui::SceneInteractionController::pointer_owner,
            D(ui, SceneInteractionController, pointer_owner)
        )
        .def(
            "has_pointer_capture",
            &ui::SceneInteractionController::has_pointer_capture,
            D(ui, SceneInteractionController, has_pointer_capture)
        )
        .def_prop_rw(
            "pointer_capture_callback",
            &ui::SceneInteractionController::pointer_capture_callback,
            &ui::SceneInteractionController::set_pointer_capture_callback,
            nb::arg().none(),
            D(ui, SceneInteractionController, pointer_capture_callback)
        )
        .def(
            "cancel_pointer_owner",
            &ui::SceneInteractionController::cancel_pointer_owner,
            D(ui, SceneInteractionController, cancel_pointer_owner)
        )
        .def(
            "handle_keyboard_event",
            &ui::SceneInteractionController::handle_keyboard_event,
            "event"_a,
            D(ui, SceneInteractionController, handle_keyboard_event)
        )
        .def(
            "handle_mouse_event",
            &ui::SceneInteractionController::handle_mouse_event,
            "event"_a,
            D(ui, SceneInteractionController, handle_mouse_event)
        )
        .def(
            "update_selection_overlay",
            &ui::SceneInteractionController::update_selection_overlay,
            D(ui, SceneInteractionController, update_selection_overlay)
        )
        .def(
            "focus_on_selection",
            &ui::SceneInteractionController::focus_on_selection,
            "camera"_a.none(),
            D(ui, SceneInteractionController, focus_on_selection)
        );

    nb::sgl_enum<ui::SceneInteractionController::PointerOwner>(
        scene_interaction_controller,
        "PointerOwner",
        D(ui, SceneInteractionController, PointerOwner)
    );
}
