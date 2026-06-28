// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/render/material/mdl/mdl_discovery.h"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <filesystem>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

/// Convert native MDL discovery records into plain Python dictionaries.
///
/// The neutral material-image schema lives in Python, so the binding deliberately
/// avoids exposing C++ schema concepts here.
nb::list
discover_mdl_materials_impl(const std::filesystem::path& library_path, const std::vector<std::string>& include_modules)
{
    nb::list records;
    for (const falcor::MDLDiscoveredMaterial& material :
         falcor::discover_mdl_materials(library_path, include_modules)) {
        nb::dict record;
        record["module_name"] = material.module_name;
        record["module_simple_name"] = material.module_simple_name;
        record["module_path"] = material.module_path;
        record["material_simple_name"] = material.material_simple_name;
        record["mdl_material_name"] = material.mdl_material_name;
        record["label"] = material.label;
        records.append(record);
    }
    return records;
}

} // namespace

FALCOR_PY_EXPORT(render_mdl_discovery)
{
    m.def(
        "discover_mdl_materials",
        &discover_mdl_materials_impl,
        "library_path"_a,
        "include_modules"_a = std::vector<std::string>(),
        "Discover MDL materials under a registered MDL search root."
    );
}
