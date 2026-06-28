// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/ui/selection_overlay.h"
#include "falcor2/render/scene.h"
#include "falcor2/render/component/camera.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>
#include <sgl/device/command.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(ui_selection_overlay)
{
    using namespace falcor;

    nb::module_ ui = nb::module_::import_("falcor2.ui");

    nb::class_<ui::SelectionOverlay, Object> selection_overlay(ui, "SelectionOverlay", D(ui, SelectionOverlay));

    nb::class_<ui::SelectionOverlay::Options>(selection_overlay, "Options", D(ui, SelectionOverlay, Options))
        .def(nb::init<>())
        .def_rw(
            "show_occluded",
            &ui::SelectionOverlay::Options::show_occluded,
            D(ui, SelectionOverlay, Options, show_occluded)
        )
        .def_rw(
            "selection_color",
            &ui::SelectionOverlay::Options::selection_color,
            D(ui, SelectionOverlay, Options, selection_color)
        )
        .def_rw(
            "fill_opacity",
            &ui::SelectionOverlay::Options::fill_opacity,
            D(ui, SelectionOverlay, Options, fill_opacity)
        );

    selection_overlay
        .def(
            nb::init<ref<sgl::Device>, std::optional<ui::SelectionOverlay::Options>>(),
            "device"_a,
            "options"_a.none() = nb::none(),
            D(ui, SelectionOverlay, SelectionOverlay)
        )
        .def_prop_rw(
            "options",
            &ui::SelectionOverlay::options,
            &ui::SelectionOverlay::set_options,
            D(ui, SelectionOverlay, options)
        )
        .def(
            "set_selected_entities",
            &ui::SelectionOverlay::set_selected_entities,
            "entities"_a,
            D(ui, SelectionOverlay, set_selected_entities)
        )
        .def(
            "set_selected_entity",
            &ui::SelectionOverlay::set_selected_entity,
            "entity"_a,
            D(ui, SelectionOverlay, set_selected_entity)
        )
        .def(
            "set_selected_geometry_instance_ids",
            &ui::SelectionOverlay::set_selected_geometry_instance_ids,
            "ids"_a,
            D(ui, SelectionOverlay, set_selected_geometry_instance_ids)
        )
        .def("clear_selection", &ui::SelectionOverlay::clear_selection, D(ui, SelectionOverlay, clear_selection))
        .def(
            "draw_overlay",
            &ui::SelectionOverlay::draw_overlay,
            "command_encoder"_a,
            "output_texture"_a,
            "geometry_instance_id_texture"_a,
            "scene"_a,
            "camera"_a,
            D(ui, SelectionOverlay, draw_overlay)
        );
}
