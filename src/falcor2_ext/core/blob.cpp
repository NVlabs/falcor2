// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "falcor2/core/blob.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(core_blob)
{
    using namespace falcor;

    nb::class_<Blob, Object>(m, "Blob", D(Blob))
        .def(
            "__init__",
            [](Blob* self, nb::bytes data, std::string name)
            {
                auto bytes = std::span(reinterpret_cast<const uint8_t*>(data.c_str()), data.size());
                new (self) Blob(bytes, std::move(name));
            },
            "data"_a,
            "name"_a = "",
            D(Blob, Blob)
        )
        .def_prop_ro(
            "data",
            [](const Blob& self)
            {
                auto data = self.data();
                return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
            },
            D(Blob, data)
        )
        .DEF_PROP_RO(Blob, name);
}
