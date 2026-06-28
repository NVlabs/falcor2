// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/utils/algorithm/parallel_reduction.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(utils_algorithm_parallel_reduction)
{
    using namespace falcor;

    nb::class_<ParallelReduction>(m, "ParallelReduction", D(ParallelReduction))
        .def(nb::init<ref<sgl::Device>>(), D(ParallelReduction, ParallelReduction))
        .def(
            "execute",
            nb::overload_cast<
                sgl::CommandEncoder*,
                sgl::DataType,
                sgl::BufferOffsetPair,
                uint32_t,
                std::optional<sgl::BufferOffsetPair>,
                std::optional<sgl::BufferOffsetPair>,
                std::optional<sgl::BufferOffsetPair>>(&ParallelReduction::execute),
            "command_encoder"_a,
            "type"_a,
            "data"_a,
            "element_count"_a,
            "sum"_a.none() = nb::none(),
            "min"_a.none() = nb::none(),
            "max"_a.none() = nb::none(),
            D(ParallelReduction, execute)
        )
        .def(
            "execute",
            nb::overload_cast<
                sgl::CommandEncoder*,
                sgl::Texture*,
                std::optional<sgl::BufferOffsetPair>,
                std::optional<sgl::BufferOffsetPair>,
                std::optional<sgl::BufferOffsetPair>>(&ParallelReduction::execute),
            "command_encoder"_a,
            "input"_a,
            "sum"_a.none() = nb::none(),
            "min"_a.none() = nb::none(),
            "max"_a.none() = nb::none(),
            D(ParallelReduction, execute)
        );

    m.def(
        "parallel_reduce",
        nb::overload_cast<
            sgl::CommandEncoder*,
            sgl::DataType,
            sgl::BufferOffsetPair,
            uint32_t,
            std::optional<sgl::BufferOffsetPair>,
            std::optional<sgl::BufferOffsetPair>,
            std::optional<sgl::BufferOffsetPair>>(&parallel_reduce),
        "command_encoder"_a,
        "type"_a,
        "data"_a,
        "element_count"_a,
        "sum"_a.none() = nb::none(),
        "min"_a.none() = nb::none(),
        "max"_a.none() = nb::none(),
        D(parallel_reduce)
    );

    m.def(
        "parallel_reduce",
        nb::overload_cast<
            sgl::CommandEncoder*,
            sgl::Texture*,
            std::optional<sgl::BufferOffsetPair>,
            std::optional<sgl::BufferOffsetPair>,
            std::optional<sgl::BufferOffsetPair>>(&parallel_reduce),
        "command_encoder"_a,
        "input"_a,
        "sum"_a.none() = nb::none(),
        "min"_a.none() = nb::none(),
        "max"_a.none() = nb::none(),
        D(parallel_reduce)
    );
}
