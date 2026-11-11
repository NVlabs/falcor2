// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/importer.h"

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"

#include <pxr/pxr.h>

#include <filesystem>

PXR_NAMESPACE_OPEN_SCOPE
class UsdStage;
PXR_NAMESPACE_CLOSE_SCOPE

namespace falcor {

/// USD importer.
class FALCOR_API UsdImporter : public Object {
    FALCOR_OBJECT(UsdImporter)
public:
    /// Load a USD scene with default import options; retains the original exported entry point.
    ref<ImporterScene> load_scene(const std::filesystem::path& path);

    /// Load a USD scene from the specified file path.
    /// @param path Path to the USD file.
    /// @param import_options Import options, including USD texture-coordinate primvar mapping.
    /// @return The imported scene.
    ref<ImporterScene> load_scene(const std::filesystem::path& path, const ImportOptions& import_options);

    /// Import a USD stage with default import options; retains the original exported entry point.
    ref<ImporterScene> import_stage(const pxr::UsdStage& stage);

    /// Import a snapshot of a caller-owned USD stage without modifying it.
    /// @param stage USD stage to import.
    /// @param import_options Import options, including USD texture-coordinate primvar mapping.
    /// @return The imported scene.
    ref<ImporterScene> import_stage(const pxr::UsdStage& stage, const ImportOptions& import_options);
};

} // namespace falcor
