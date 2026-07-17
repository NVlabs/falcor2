// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "falcor2/core/asset_resolver.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(core_asset_resolver)
{
    using namespace falcor;

    nb::class_<AssetResolver, Object>(m, "AssetResolver")
        .def(nb::init<>())
        .def(nb::init<const std::string&>(), "paths"_a, "Create a resolver from a semicolon-separated path list.")
        .def(nb::init<const std::filesystem::path&>(), "path"_a, "Create a resolver from one literal path.")
        .def(
            nb::init<std::span<const std::filesystem::path>>(),
            "paths"_a,
            "Create a resolver from literal paths in the supplied order."
        )
        .def(
            "push",
            nb::overload_cast<const std::string&>(&AssetResolver::push, nb::const_),
            "paths"_a,
            "Return a new resolver with a semicolon-separated path list at highest priority."
        )
        .def(
            "push",
            nb::overload_cast<const std::filesystem::path&>(&AssetResolver::push, nb::const_),
            "path"_a,
            "Return a new resolver with one literal path at highest priority."
        )
        .def(
            "push",
            nb::overload_cast<std::span<const std::filesystem::path>>(&AssetResolver::push, nb::const_),
            "paths"_a,
            "Return a new resolver with the literal paths at highest priority."
        )
        .def_prop_ro(
            "search_paths",
            &AssetResolver::get_search_paths,
            "Explicit search roots in effective priority order."
        )
        .def(
            "resolve_path",
            &AssetResolver::resolve_path,
            "asset_path"_a,
            "required"_a = false,
            "Resolve a single path. If required is true, raise RuntimeError when it cannot be found."
        )
        .def(
            "resolve_path_pattern",
            &AssetResolver::resolve_path_pattern,
            "directory_path"_a,
            "filename_pattern"_a,
            "first_match_only"_a = true,
            "Resolve a filename regular expression in the first candidate directory containing matches."
        );
}
