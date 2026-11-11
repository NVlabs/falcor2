// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/falcor2.h"
#include "falcor2/utils/buffer_handle.h"

#include <sgl/device/resource.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(utils_buffer_handle)
{
    using namespace falcor;

    nb::class_<BufferHandle>(m, "BufferHandle", D(BufferHandle))
        .def_prop_ro("data", &BufferHandle::data, D(BufferHandle, data));

    nb::class_<RWBufferHandle>(m, "RWBufferHandle", D(RWBufferHandle))
        .def_prop_ro("data", &RWBufferHandle::data, D(RWBufferHandle, data));

    m.def(
        "to_handle",
        [](const ref<sgl::Buffer>& buffer)
        {
            return to_handle(buffer);
        },
        "buffer"_a
    );
    m.def("to_rw_handle", &to_rw_handle, "buffer"_a, D(to_rw_handle));
}
