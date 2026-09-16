// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/fwd.h"
#include "falcor2/core/macros.h"
#include "falcor2/importers/fwd.h"

namespace falcor {

/// Generate tangents using MikkTSpace, splitting vertices where its per-corner output is discontinuous.
/// Existing vertices are never merged; exact compaction is a separate operation that may run before or after this.
FALCOR_API void mikkt_generate_tangent_space(ImporterMesh& mesh);

/// Generate tangents using cached per-corner output, splitting vertices and rewriting indices as above.
/// On a cache miss, the per-corner tangents are computed and stored in the cache.
FALCOR_API void mikkt_generate_tangent_space(ImporterMesh& mesh, BlobCache& cache);

} // namespace falcor
