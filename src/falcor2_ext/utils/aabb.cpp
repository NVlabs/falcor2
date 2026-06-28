// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/utils/aabb.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(utils_aabb)
{
    using namespace falcor;

    nb::class_<AABB>(m, "AABB", D(AABB))
        .def(nb::init<>(), D(AABB, AABB))
        .def(nb::init<float3, float3>(), "min"_a, "max"_a, D(AABB, AABB, 2))
        .DEF_RW(AABB, min)
        .DEF_RW(AABB, max)
        .def("is_valid", &AABB::is_valid, D(AABB, is_valid))
        .def("expand", nb::overload_cast<const float3&>(&AABB::expand), "point"_a, D(AABB, expand))
        .def(
            "expand",
            nb::overload_cast<const float3&, float>(&AABB::expand),
            "point"_a,
            "radius"_a,
            D(AABB, expand, 2)
        )
        .def("expand", nb::overload_cast<const AABB&>(&AABB::expand), "other"_a, D(AABB, expand, 3))
        .def("transform", &AABB::transform, "matrix"_a, D(AABB, transform))
        .DEF_PROP_RO(AABB, center)
        .DEF_PROP_RO(AABB, size)
        .def(
            "__repr__",
            [](const AABB& aabb)
            {
                if (!aabb.is_valid()) {
                    return std::string("<AABB invalid>");
                }
                return "<AABB min=(" + std::to_string(aabb.min.x) + ", " + std::to_string(aabb.min.y) + ", "
                    + std::to_string(aabb.min.z) + ") max=(" + std::to_string(aabb.max.x) + ", "
                    + std::to_string(aabb.max.y) + ", " + std::to_string(aabb.max.z) + ")>";
            }
        );
}
