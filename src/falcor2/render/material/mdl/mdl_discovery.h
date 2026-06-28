// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"

#include <filesystem>
#include <string>
#include <vector>

namespace falcor {

/// Description of one renderable MDL material found below a library search root.
///
/// `mdl_material_name` is the fully qualified MDL material name, including the
/// overload signature when the MDL SDK reports one. This is the value passed to
/// `MDLMaterial` as its `mdl_material_name` property.
struct MDLDiscoveredMaterial {
    /// Module name without the leading MDL scope marker.
    std::string module_name;
    /// Final module component, useful for display and filtering.
    std::string module_simple_name;
    /// Resolved filesystem path of the source module.
    std::string module_path;
    /// Material name without module qualification.
    std::string material_simple_name;
    /// Fully qualified MDL material name suitable for `MDLMaterial`.
    std::string mdl_material_name;
    /// Human-readable label for manifests and reports.
    std::string label;
};

/// Discover MDL materials below `library_path`.
///
/// The implementation reuses the process-wide `MDLContext`, temporarily adds
/// `library_path` as an MDL search path, calls the MDL SDK discovery API, loads
/// selected modules, and enumerates their material definitions. Returned modules
/// are filtered to files contained by `library_path`. `include_modules` may be
/// empty, or may contain simple/qualified module names or substrings used to
/// keep only matching modules.
FALCOR_API std::vector<MDLDiscoveredMaterial>
discover_mdl_materials(const std::filesystem::path& library_path, const std::vector<std::string>& include_modules);

} // namespace falcor
