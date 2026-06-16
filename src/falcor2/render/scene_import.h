// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fwd.h"

#include "falcor2/importers/fwd.h"
#include "falcor2/core/macros.h"

#include <filesystem>

namespace falcor {

/// Load an importer scene.
/// @param scene Scene to populate.
/// @param importer_scene Importer scene to load from.
FALCOR_API void load_importer_scene(Scene* scene, const ImporterScene& importer_scene);

/// Load a scene from disk.
/// @param scene Scene to populate.
/// @param path Path to scene file.
FALCOR_API void load_scene(Scene* scene, const std::filesystem::path& path, bool recompute_normals);

} // namespace falcor
