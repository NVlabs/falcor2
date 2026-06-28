// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/denoisers/dlss_g_feature.h"
#include "falcor2/denoisers/dlss_rr_feature.h"
#include "falcor2/denoisers/dlss_sr_feature.h"
#include "falcor2/denoisers/ngx.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(denoisers_ngx_features)
{
    using namespace falcor;
    using namespace falcor::ngx;

    nb::module_ ngx_module = nb::module_::import_("falcor2.ngx");

    nb::class_<DLSSRRFeature, Object>(ngx_module, "DLSSRRFeature", D(ngx, DLSSRRFeature))
        .def_prop_ro("device", &DLSSRRFeature::device, D(ngx, DLSSRRFeature, device))
        .def_prop_ro("ngx", &DLSSRRFeature::ngx, D(ngx, DLSSRRFeature, ngx))
        .def_prop_ro("desc", &DLSSRRFeature::desc, nb::rv_policy::reference_internal, D(ngx, DLSSRRFeature, desc))
        .def_prop_ro(
            "last_evaluate_result",
            &DLSSRRFeature::last_evaluate_result,
            D(ngx, DLSSRRFeature, last_evaluate_result)
        )
        .def(
            "evaluate",
            nb::overload_cast<const DLSSRREvaluateDesc&>(&DLSSRRFeature::evaluate),
            "desc"_a,
            D(ngx, DLSSRRFeature, evaluate)
        )
        .def(
            "evaluate",
            nb::overload_cast<sgl::CommandEncoder*, const DLSSRREvaluateDesc&>(&DLSSRRFeature::evaluate),
            "command_encoder"_a,
            "desc"_a,
            D(ngx, DLSSRRFeature, evaluate)
        );

    nb::class_<DLSSSRFeature, Object>(ngx_module, "DLSSSRFeature", D_NA(ngx, DLSSSRFeature))
        .def_prop_ro("device", &DLSSSRFeature::device, D_NA(ngx, DLSSSRFeature, device))
        .def_prop_ro("ngx", &DLSSSRFeature::ngx, D_NA(ngx, DLSSSRFeature, ngx))
        .def_prop_ro("desc", &DLSSSRFeature::desc, nb::rv_policy::reference_internal, D_NA(ngx, DLSSSRFeature, desc))
        .def_prop_ro(
            "last_evaluate_result",
            &DLSSSRFeature::last_evaluate_result,
            D_NA(ngx, DLSSSRFeature, last_evaluate_result)
        )
        .def(
            "evaluate",
            nb::overload_cast<const DLSSSREvaluateDesc&>(&DLSSSRFeature::evaluate),
            "desc"_a,
            D_NA(ngx, DLSSSRFeature, evaluate)
        )
        .def(
            "evaluate",
            nb::overload_cast<sgl::CommandEncoder*, const DLSSSREvaluateDesc&>(&DLSSSRFeature::evaluate),
            "command_encoder"_a,
            "desc"_a,
            D_NA(ngx, DLSSSRFeature, evaluate)
        );

    nb::class_<DLSSGFeature, Object>(ngx_module, "DLSSGFeature", D_NA(ngx, DLSSGFeature))
        .def_prop_ro("device", &DLSSGFeature::device, D_NA(ngx, DLSSGFeature, device))
        .def_prop_ro("ngx", &DLSSGFeature::ngx, D_NA(ngx, DLSSGFeature, ngx))
        .def_prop_ro("desc", &DLSSGFeature::desc, nb::rv_policy::reference_internal, D_NA(ngx, DLSSGFeature, desc))
        .def_prop_ro(
            "last_evaluate_result",
            &DLSSGFeature::last_evaluate_result,
            D_NA(ngx, DLSSGFeature, last_evaluate_result)
        )
        .def(
            "evaluate",
            nb::overload_cast<const DLSSGEvaluateDesc&>(&DLSSGFeature::evaluate),
            "desc"_a,
            D_NA(ngx, DLSSGFeature, evaluate)
        )
        .def(
            "evaluate",
            nb::overload_cast<sgl::CommandEncoder*, const DLSSGEvaluateDesc&>(&DLSSGFeature::evaluate),
            "command_encoder"_a,
            "desc"_a,
            D_NA(ngx, DLSSGFeature, evaluate)
        );
}
