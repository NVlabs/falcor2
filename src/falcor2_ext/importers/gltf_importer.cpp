// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "falcor2/importers/gltf_importer/gltf_importer.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(importers_gltf_importer)
{
    using namespace falcor;

    nb::class_<GltfImporter, Object>(m, "GltfImporter", D(GltfImporter))
        .def(nb::init<>())
        .def("load_scene", &GltfImporter::load_scene, "path"_a, D(GltfImporter, load_scene));
}
