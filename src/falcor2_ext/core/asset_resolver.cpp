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
        .def("add_search_path", &AssetResolver::add_search_path, "path"_a, "Adds search path to the AssetResolver.")
        .def(
            "add_search_paths",
            &AssetResolver::add_search_paths,
            "paths"_a,
            "Adds semicolon-separated search paths to the AssetResolver."
        )
        .def("resolve_path", &AssetResolver::resolve_path, "asset_path"_a, "Resolves a single path.")
        .def("clone", &AssetResolver::clone, "Deep clones the asset resolver")
        .def("shallow_clone", &AssetResolver::shallow_clone, "Shallow clones the asset resolver")
        .def(
            "resolve_path_pattern",
            &AssetResolver::resolve_path_pattern,
            "directory_path"_a,
            "filename_pattern"_a,
            "first_match_only"_a = true,
            "Resolves directory_path/filename_pattern paths. Filename_pattern is a regular expression. Resolves all or "
            "just first that matches. Used for UDIM."
        );
}
