// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "nbdictionary.h"

#include "falcor2/falcor2.h"
#include "falcor2/utils/buffer_handle.h"
#include "falcor2/utils/managed_buffer.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>
#include <sgl/device/command.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(utils_managed_buffer)
{
    using namespace falcor;

    nb::class_<BufferHandle>(m, "BufferHandle", D(BufferHandle))
        .def_prop_ro("data", &BufferHandle::data, D(BufferHandle, data))
        .def(
            "get_uniforms",
            [](BufferHandle* self)
            {
                return NBDictionary::get_uniforms(*self);
            }
        )
        .def(
            "get_this",
            [](BufferHandle* self)
            {
                auto dict = NBDictionary::get_uniforms(*self);
                dict["_type"] = "BufferHandle";
                return dict;
            }
        );

    m.def(
        "to_handle",
        [](const ref<sgl::Buffer>& buffer)
        {
            return to_handle(buffer);
        },
        "buffer"_a
    );

    nb::class_<ManagedBuffer, Object>(m, "ManagedBuffer", D(ManagedBuffer))
        .def(nb::init<sgl::BufferDesc>(), "desc"_a, D(ManagedBuffer, ManagedBuffer))
        .def("size", &ManagedBuffer::size, D(ManagedBuffer, size))
        .def(
            "set_data",
            [](ManagedBuffer* self, nb::bytes src, size_t offset)
            {
                self->set_data(src.c_str(), src.size(), offset);
            },
            "src"_a,
            "offset"_a = 0,
            D(ManagedBuffer, set_data)
        )
        .def("resize", &ManagedBuffer::resize, "size"_a, D(ManagedBuffer, resize))
        .def("reset_dirty_range", &ManagedBuffer::reset_dirty_range, D(ManagedBuffer, reset_dirty_range))
        .def("is_dirty", &ManagedBuffer::is_dirty, D(ManagedBuffer, is_dirty))
        .def("mark_dirty", &ManagedBuffer::mark_dirty, "begin"_a, "end"_a, D(ManagedBuffer, mark_dirty))
        .def("mark_all_dirty", &ManagedBuffer::mark_all_dirty, D(ManagedBuffer, mark_all_dirty))
        .def(
            "update_buffer",
            nb::overload_cast<sgl::Device*>(&ManagedBuffer::update_buffer),
            "device"_a,
            D(ManagedBuffer, update_buffer)
        )
        .def(
            "update_buffer",
            nb::overload_cast<sgl::CommandEncoder*>(&ManagedBuffer::update_buffer),
            "command_encoder"_a,
            D(ManagedBuffer, update_buffer)
        )
        .def_prop_ro("buffer", &ManagedBuffer::buffer, D(ManagedBuffer, buffer));

    m.def(
        "to_handle",
        [](const ref<ManagedBuffer>& managed_buffer)
        {
            return to_handle(managed_buffer);
        },
        "managed_buffer"_a
    );
}
