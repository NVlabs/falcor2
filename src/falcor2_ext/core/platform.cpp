// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/core/platform.h"

FALCOR_PY_EXPORT(core_platform)
{
    using namespace falcor::platform;

    nb::module_ platform = nb::module_::import_("falcor2.platform");

    nb::class_<ScopedGPURealtimePriority>(
        platform,
        "_ScopedGPURealtimePriority",
        D(platform, ScopedGPURealtimePriority)
    )
        .def("close", &ScopedGPURealtimePriority::close, D(platform, ScopedGPURealtimePriority, close))
        .def(
            "__enter__",
            [](ScopedGPURealtimePriority& self) -> ScopedGPURealtimePriority&
            {
                return self;
            },
            nb::rv_policy::reference_internal
        )
        .def(
            "__exit__",
            [](ScopedGPURealtimePriority& self, nb::handle, nb::handle, nb::handle)
            {
                self.close();
            },
            nb::arg().none(),
            nb::arg().none(),
            nb::arg().none()
        );

    platform.def(
        "set_gpu_realtime_priority",
        []()
        {
            return new ScopedGPURealtimePriority();
        },
        nb::rv_policy::take_ownership,
        D(platform, ScopedGPURealtimePriority)
    );

    platform.def("project_directory", &project_directory, D(platform, project_directory));
    platform.def("runtime_directory", &runtime_directory, D(platform, runtime_directory));
    platform.def("app_data_directory", &app_data_directory, D(platform, app_data_directory));
}
