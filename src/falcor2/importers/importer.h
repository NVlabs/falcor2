// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/fwd.h"
#include "falcor2/core/types.h"
#include "falcor2/core/object.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/properties.h"
#include "falcor2/importers/fwd.h"

namespace falcor {

struct ImportOptions {
    bool recompute_normals = false;
};

ref<ImporterScene> FALCOR_API import_scene(const std::filesystem::path& path, const ImportOptions& import_options = {});

/// Set the global blob cache used by importers (e.g. for MikkTSpace caching).
FALCOR_API void set_importer_cache(ref<BlobCache> cache);

/// Get the global blob cache used by importers. May return nullptr if not set.
FALCOR_API ref<BlobCache> importer_cache();

} // namespace falcor
