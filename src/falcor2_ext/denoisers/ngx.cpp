// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/denoisers/dlss_g_feature.h"
#include "falcor2/denoisers/dlss_rr_feature.h"
#include "falcor2/denoisers/dlss_sr_feature.h"
#include "falcor2/denoisers/ngx.h"

#include <sgl/device/device.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(denoisers_ngx)
{
    using namespace falcor;
    using namespace falcor::ngx;

    nb::module_ ngx_module = nb::module_::import_("falcor2.ngx");

    nb::class_<NGX, Object>(ngx_module, "NGX", D(ngx, NGX))
        .def_static(
            "get",
            [](sgl::Device* device)
            {
                return NGX::get(device);
            },
            "device"_a,
            D(ngx, NGX, get)
        )
        .def_prop_ro("device", &NGX::device, D(ngx, NGX, device))
        .def_prop_ro("info", &NGX::info, nb::rv_policy::reference_internal, D(ngx, NGX, info))
        .def(
            "get_dlssd_optimal_settings",
            &NGX::get_dlssd_optimal_settings,
            "target_width"_a,
            "target_height"_a,
            "quality"_a = QualityMode::quality,
            D(ngx, NGX, get_dlssd_optimal_settings)
        )
        .def(
            "get_dlss_optimal_settings",
            &NGX::get_dlss_optimal_settings,
            "target_width"_a,
            "target_height"_a,
            "quality"_a = QualityMode::quality,
            D_NA(ngx, NGX, get_dlss_optimal_settings)
        )
        .def("create_dlss_rr_feature", &NGX::create_dlss_rr_feature, "desc"_a, D(ngx, NGX, create_dlss_rr_feature))
        .def("create_dlss_sr_feature", &NGX::create_dlss_sr_feature, "desc"_a, D_NA(ngx, NGX, create_dlss_sr_feature))
        .def("create_dlss_g_feature", &NGX::create_dlss_g_feature, "desc"_a, D_NA(ngx, NGX, create_dlss_g_feature));

    ngx_module.def("get_vulkan_pre_device_info", &get_vulkan_pre_device_info, D(ngx, get_vulkan_pre_device_info));
}
