// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/ui/scene_editor.h"
#include "falcor2/ui/camera_controller.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/scene.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(ui_scene_editor)
{
    using namespace falcor;

    nb::module_ ui = nb::module_::import_("falcor2.ui");

    nb::class_<ui::SceneEditor, Object> scene_editor(ui, "SceneEditor", D(ui, SceneEditor));
    scene_editor //
        .def(nb::init<>(), D(ui, SceneEditor, SceneEditor))
        .def_prop_rw(
            "scene",
            &ui::SceneEditor::scene,
            &ui::SceneEditor::set_scene,
            nb::arg().none(),
            D(ui, SceneEditor, scene)
        )
        .def_prop_rw(
            "camera_controller",
            &ui::SceneEditor::camera_controller,
            &ui::SceneEditor::set_camera_controller,
            nb::arg().none(),
            D(ui, SceneEditor, camera_controller)
        )
        .def_prop_rw(
            "selected_object",
            &ui::SceneEditor::selected_object,
            &ui::SceneEditor::set_selected_object,
            nb::arg().none(),
            D(ui, SceneEditor, selected_object)
        )
        .def(
            "remove_selected_object",
            &ui::SceneEditor::remove_selected_object,
            D(ui, SceneEditor, remove_selected_object)
        )
        .def_prop_rw(
            "tool_mode",
            &ui::SceneEditor::tool_mode,
            &ui::SceneEditor::set_tool_mode,
            D(ui, SceneEditor, tool_mode)
        )
        .def_prop_rw(
            "transform_space",
            &ui::SceneEditor::transform_space,
            &ui::SceneEditor::set_transform_space,
            D(ui, SceneEditor, transform_space)
        )
        .def(
            "set_active_camera",
            &ui::SceneEditor::set_active_camera,
            "camera"_a.none(),
            D(ui, SceneEditor, set_active_camera)
        )
        .def(
            "set_camera_matrices",
            &ui::SceneEditor::set_camera_matrices,
            "view"_a,
            "proj"_a,
            D(ui, SceneEditor, set_camera_matrices)
        )
        .def_prop_rw("playing", &ui::SceneEditor::playing, &ui::SceneEditor::set_playing, D(ui, SceneEditor, playing))
        .def_prop_rw("looping", &ui::SceneEditor::looping, &ui::SceneEditor::set_looping, D(ui, SceneEditor, looping))
        .def("update_playback", &ui::SceneEditor::update_playback, "dt"_a, D(ui, SceneEditor, update_playback))
        .def_prop_rw("visible", &ui::SceneEditor::visible, &ui::SceneEditor::set_visible, D(ui, SceneEditor, visible))
        .def_prop_ro("viewport_state", &ui::SceneEditor::viewport_state, D(ui, SceneEditor, viewport_state))
        .def_prop_ro("selection_version", &ui::SceneEditor::selection_version, D(ui, SceneEditor, selection_version))
        .def("editor_ui", &ui::SceneEditor::editor_ui, D(ui, SceneEditor, editor_ui))
        .def(
            "viewport_ui",
            &ui::SceneEditor::viewport_ui,
            "output_texture"_a.none(),
            "fps"_a,
            D(ui, SceneEditor, viewport_ui)
        )
        .def(
            "is_viewport_interactive",
            &ui::SceneEditor::is_viewport_interactive,
            D(ui, SceneEditor, is_viewport_interactive)
        )
        .def(
            "can_pick_at",
            [](const ui::SceneEditor& self, float2 screen_pos) -> nb::object
            {
                uint2 local_pos;
                if (self.can_pick_at(screen_pos, local_pos))
                    return nb::cast(local_pos);
                return nb::none();
            },
            "screen_pos"_a,
            D(ui, SceneEditor, can_pick_at)
        )
        .def(
            "add_help_section",
            [](ui::SceneEditor& self,
               const std::string& title,
               const std::vector<std::pair<std::string, std::string>>& entries)
            {
                ui::SceneEditor::HelpSection section;
                section.title = title;
                section.entries.reserve(entries.size());
                for (const auto& [key, desc] : entries)
                    section.entries.push_back({key, desc});
                self.add_help_section(std::move(section));
            },
            "title"_a,
            "entries"_a,
            D(ui, SceneEditor, add_help_section)
        );

    nb::sgl_enum<ui::SceneEditor::ToolMode>(scene_editor, "ToolMode", D(ui, SceneEditor, ToolMode));
    nb::sgl_enum<ui::SceneEditor::TransformSpace>(scene_editor, "TransformSpace", D(ui, SceneEditor, TransformSpace));

    nb::class_<ui::SceneEditor::ViewportState>(scene_editor, "ViewportState", D(ui, SceneEditor, ViewportState))
        .def_ro("pos", &ui::SceneEditor::ViewportState::pos, D(ui, SceneEditor, ViewportState, pos))
        .def_ro("size", &ui::SceneEditor::ViewportState::size, D(ui, SceneEditor, ViewportState, size))
        .def_ro("hovered", &ui::SceneEditor::ViewportState::hovered, D(ui, SceneEditor, ViewportState, hovered))
        .def_ro("focused", &ui::SceneEditor::ViewportState::focused, D(ui, SceneEditor, ViewportState, focused))
        .def_ro("valid", &ui::SceneEditor::ViewportState::valid, D(ui, SceneEditor, ViewportState, valid));
}
