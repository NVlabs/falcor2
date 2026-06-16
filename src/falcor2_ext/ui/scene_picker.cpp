// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/ui/scene_picker.h"
#include "falcor2/render/scene.h"
#include "falcor2/render/component/camera.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>
#include <sgl/device/command.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(ui_scene_picker)
{
    using namespace falcor;

    nb::module_ ui = nb::module_::import_("falcor2.ui");

    nb::class_<ui::ScenePicker, Object>(ui, "ScenePicker", D(ui, ScenePicker))
        .def(nb::init<ref<sgl::Device>>(), "device"_a, D(ui, ScenePicker, ScenePicker))
        .def("render", &ui::ScenePicker::render, "command_encoder"_a, "scene"_a, "camera"_a, D(ui, ScenePicker, render))
        .def_prop_ro(
            "geometry_instance_id_texture",
            &ui::ScenePicker::geometry_instance_id_texture,
            D(ui, ScenePicker, geometry_instance_id_texture)
        )
        .def(
            "pick",
            &ui::ScenePicker::pick,
            "position"_a,
            "geometry_instance_id_texture"_a = nb::none(),
            D(ui, ScenePicker, pick)
        )
        .def(
            "pick_entity",
            &ui::ScenePicker::pick_entity,
            "scene"_a,
            "position"_a,
            "geometry_instance_id_texture"_a = nb::none(),
            D(ui, ScenePicker, pick_entity)
        )
        .def_static(
            "find_entity_by_geometry_instance_id",
            &ui::ScenePicker::find_entity_by_geometry_instance_id,
            "scene"_a,
            "geometry_instance_id"_a,
            D(ui, ScenePicker, find_entity_by_geometry_instance_id)
        );
}
