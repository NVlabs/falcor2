// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/fwd.h"
#include "falcor2/core/macros.h"
#include "falcor2/importers/fwd.h"

namespace falcor {

/// Generate tangents for mesh vertices using MikkTSpace algorithm.
FALCOR_API void mikkt_generate_tangent_space(ImporterMesh& mesh);

/// Generate tangents with caching. On cache hit, tangents are read from cache.
/// On cache miss, tangents are computed and stored in cache.
FALCOR_API void mikkt_generate_tangent_space(ImporterMesh& mesh, BlobCache& cache);

} // namespace falcor
