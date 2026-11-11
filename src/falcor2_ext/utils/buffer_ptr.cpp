// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "falcor2/utils/buffer_ptr.h"

using namespace nb::literals;

template<typename BufferPtr>
static void bind_buffer_ptr(nb::module_& m, const char* name)
{
    // Both public aliases use documentation from the shared implementation.
    nb::class_<BufferPtr>(m, name, D(detail, BufferPtr))
        .def(nb::init<>())
        .def_prop_ro("data", &BufferPtr::data, D(detail, BufferPtr, data))
        .def_prop_ro("is_valid", &BufferPtr::is_valid, D(detail, BufferPtr, is_valid));
}

FALCOR_PY_EXPORT(utils_buffer_ptr)
{
    using namespace falcor;
    bind_buffer_ptr<BufferPtr>(m, "BufferPtr");
    bind_buffer_ptr<RWBufferPtr>(m, "RWBufferPtr");
    m.def(
        "to_buffer_ptr",
        &to_buffer_ptr,
        "buffer"_a,
        "byte_offset"_a = 0,
        "alignment"_a = 4,
        D(detail, BufferPtr, from_buffer)
    );
    m.def(
        "to_rw_buffer_ptr",
        &to_rw_buffer_ptr,
        "buffer"_a,
        "byte_offset"_a = 0,
        "alignment"_a = 4,
        D(detail, BufferPtr, from_buffer)
    );
}
