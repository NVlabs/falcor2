// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/importer_types.h"

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"

#include <filesystem>
#include <initializer_list>
#include <string_view>

namespace falcor {

/// USD importer.
class FALCOR_API UsdImporter : public Object {
    FALCOR_OBJECT(UsdImporter)
public:
    /// Load a USD scene from the specified file path.
    /// @param path Path to the USD file.
    /// @return The imported scene.
    ref<ImporterScene> load_scene(const std::filesystem::path& path);
};

} // namespace falcor
