// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        .def(
            "load_scene",
            nb::overload_cast<const std::filesystem::path&, const ImportOptions&>(&GltfImporter::load_scene),
            "path"_a,
            "import_options"_a = ImportOptions(),
            "Load a glTF scene with optional import options."
        );
}
