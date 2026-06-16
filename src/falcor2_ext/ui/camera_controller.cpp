// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/ui/camera_controller.h"
#include "falcor2/render/transform.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(ui_camera_controller)
{
    using namespace falcor;

    nb::module_ ui = nb::module_::import_("falcor2.ui");

    nb::class_<ui::CameraController, Object> camera_controller(ui, "CameraController", D(ui, CameraController));
    camera_controller //
        .def(nb::init<>(), D(ui, CameraController, CameraController))
        .def_prop_rw(
            "transform",
            &ui::CameraController::transform,
            &ui::CameraController::set_transform,
            D(ui, CameraController, transform)
        )
        .def_prop_rw(
            "pivot",
            &ui::CameraController::pivot,
            &ui::CameraController::set_pivot,
            D(ui, CameraController, pivot)
        )
        .def_prop_rw(
            "orbit_distance",
            &ui::CameraController::orbit_distance,
            &ui::CameraController::set_orbit_distance,
            D(ui, CameraController, orbit_distance)
        )
        .def_prop_rw(
            "move_speed",
            &ui::CameraController::move_speed,
            &ui::CameraController::set_move_speed,
            D(ui, CameraController, move_speed)
        )
        .def_prop_rw(
            "smoothing",
            &ui::CameraController::smoothing,
            &ui::CameraController::set_smoothing,
            D(ui, CameraController, smoothing)
        )
        .def_prop_rw(
            "capture_callback",
            &ui::CameraController::capture_callback,
            &ui::CameraController::set_capture_callback,
            nb::arg().none(),
            D(ui, CameraController, capture_callback)
        )
        .def("is_captured", &ui::CameraController::is_captured, D(ui, CameraController, is_captured))
        .def_prop_ro("state", &ui::CameraController::state, D(ui, CameraController, state))
        .def("focus", &ui::CameraController::focus, "target"_a, "distance"_a, D(ui, CameraController, focus))
        .def(
            "handle_keyboard_event",
            &ui::CameraController::handle_keyboard_event,
            "event"_a,
            D(ui, CameraController, handle_keyboard_event)
        )
        .def(
            "handle_mouse_event",
            &ui::CameraController::handle_mouse_event,
            "event"_a,
            D(ui, CameraController, handle_mouse_event)
        )
        .def("update", &ui::CameraController::update, "dt"_a, D(ui, CameraController, update));

    nb::sgl_enum<ui::CameraController::State>(camera_controller, "State", D(ui, CameraController, State));
}
