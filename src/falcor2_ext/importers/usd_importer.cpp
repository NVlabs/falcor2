// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "falcor2/importers/usd_importer/usd_importer.h"
#include "render/usd_stage_bridge.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(importers_usd_importer)
{
    using namespace falcor;

    nb::class_<UsdImporter, Object>(m, "UsdImporter", D(UsdImporter))
        .def(nb::init<>())
        .def(
            "load_scene",
            nb::overload_cast<const std::filesystem::path&, const ImportOptions&>(&UsdImporter::load_scene),
            "path"_a,
            "import_options"_a = ImportOptions(),
            D(UsdImporter, load_scene, 2)
        )
        .def(
            "import_stage",
            [](UsdImporter& self, nb::object usd_stage, const ImportOptions& import_options)
            {
                const pxr::UsdStage* stage = usd_stage_bridge::try_extract(usd_stage.ptr());
                if (!stage)
                    throw nb::type_error("stage must be a falcor2.pxr.Usd.Stage");
                return self.import_stage(*stage, import_options);
            },
            "stage"_a,
            "import_options"_a = ImportOptions(),
            nb::sig(
                "def import_stage(self, stage: falcor2.pxr.Usd.Stage, import_options: ImportOptions = ...) -> "
                "ImporterScene"
            ),
            D(UsdImporter, import_stage, 2)
        );
}
