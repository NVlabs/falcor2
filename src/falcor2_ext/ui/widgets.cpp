// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/ui/widgets.h"

#include "falcor2/render/transform.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(ui_widgets)
{
    using namespace falcor;

    nb::module_ ui = nb::module_::import_("falcor2.ui");

    nb::class_<ui::ViewportOverlayDesc>(ui, "ViewportOverlayDesc", D(ui, ViewportOverlayDesc))
        .def(nb::init<>())
        .def_rw("screen_pos", &ui::ViewportOverlayDesc::screen_pos, D(ui, ViewportOverlayDesc, screen_pos))
        .def_rw("screen_size", &ui::ViewportOverlayDesc::screen_size, D(ui, ViewportOverlayDesc, screen_size))
        .def_rw("fps", &ui::ViewportOverlayDesc::fps, D(ui, ViewportOverlayDesc, fps));

    ui.def("draw_viewport_overlay", &ui::draw_viewport_overlay, "desc"_a, D(ui, draw_viewport_overlay));
    ui.def(
        "draw_viewport_overlay",
        [](float2 screen_pos, float2 screen_size, float fps)
        {
            ui::draw_viewport_overlay({
                .screen_pos = screen_pos,
                .screen_size = screen_size,
                .fps = fps,
            });
        },
        "screen_pos"_a,
        "screen_size"_a,
        "fps"_a,
        D(ui, draw_viewport_overlay)
    );

    ui.def("transform_editor", &ui::transform_editor, "label"_a, "transform"_a, D(ui, transform_editor));
}
