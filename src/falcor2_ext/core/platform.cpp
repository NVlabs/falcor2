// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/core/platform.h"

FALCOR_PY_EXPORT(core_platform)
{
    using namespace falcor::platform;

    nb::module_ platform = nb::module_::import_("falcor2.platform");

    platform.def("project_directory", &project_directory, D(platform, project_directory));
    platform.def("runtime_directory", &runtime_directory, D(platform, runtime_directory));
    platform.def("app_data_directory", &app_data_directory, D(platform, app_data_directory));
}
