// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "nanobind_reflector.h"

#include "falcor2/render/reconstruction_filter.h"

FALCOR_PY_EXPORT(render_reconstruction_filter)
{
    using namespace falcor;

    nb::class_<ReconstructionFilter, ReflectedObject> base(m, "ReconstructionFilter", D(ReconstructionFilter));
    reflection::bind<ReconstructionFilter>(base);
    base.DEF_PROP_RO(ReconstructionFilter, slang_type_name)
        .DEF_PROP_RO(ReconstructionFilter, generation)
        .def("notify_changed", &ReconstructionFilter::notify_changed, D(ReconstructionFilter, notify_changed))
        .def(
            "write_to_cursor",
            nb::overload_cast<sgl::ShaderCursor>(&ReconstructionFilter::write_to_cursor, nb::const_),
            "cursor"_a,
            D(ReconstructionFilter, write_to_cursor, 2)
        );

    nb::class_<BoxReconstructionFilter, ReconstructionFilter> box(
        m,
        "BoxReconstructionFilter",
        D(BoxReconstructionFilter)
    );
    reflection::bind<BoxReconstructionFilter>(box);
    box.def(nb::init<float>(), "width"_a = 1.f, D(BoxReconstructionFilter, BoxReconstructionFilter));

    nb::class_<GaussianReconstructionFilter, ReconstructionFilter> gaussian(
        m,
        "GaussianReconstructionFilter",
        D(GaussianReconstructionFilter)
    );
    reflection::bind<GaussianReconstructionFilter>(gaussian);
    gaussian.def(
        nb::init<float, float>(),
        "sigma"_a = 0.5f,
        "radius"_a = 1.5f,
        D(GaussianReconstructionFilter, GaussianReconstructionFilter)
    );
}
