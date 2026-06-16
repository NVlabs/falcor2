// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/utils/algorithm/prefix_sum.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(utils_algorithm_prefix_sum)
{
    using namespace falcor;

    nb::class_<PrefixSum>(m, "PrefixSum", D(PrefixSum))
        .def(nb::init<ref<sgl::Device>>(), D(PrefixSum, PrefixSum))
        .def(
            "execute",
            &PrefixSum::execute,
            "command_encoder"_a,
            "type"_a,
            "data"_a,
            "element_count"_a,
            "total_sum"_a.none() = nb::none(),
            D(PrefixSum, execute)
        );

    m.def(
        "prefix_sum",
        &prefix_sum,
        "command_encoder"_a,
        "type"_a,
        "data"_a,
        "element_count"_a,
        "total_sum"_a.none() = nb::none(),
        D(prefix_sum)
    );
}
