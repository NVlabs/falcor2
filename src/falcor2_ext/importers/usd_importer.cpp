// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "falcor2/importers/usd_importer/usd_importer.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(importers_usd_importer)
{
    using namespace falcor;

    nb::class_<UsdImporter, Object>(m, "UsdImporter", D(UsdImporter))
        .def(nb::init<>())
        .def("load_scene", &UsdImporter::load_scene, "path"_a, D(UsdImporter, load_scene));
}
