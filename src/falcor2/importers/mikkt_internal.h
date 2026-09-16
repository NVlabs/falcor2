// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/types.h"
#include "falcor2/importers/fwd.h"

#include <vector>

#if defined(FALCOR_ENABLE_MIKKT_COMPARISON)
namespace falcor::mikkt_detail {

/// Tangent generators retained for internal correctness comparison.
enum class Generator {
    official_mikktspace,
    parallel_mikktspace,
};

/// Generate per-corner tangent space using the comparison-only parallel implementation.
std::vector<float4> generate_parallel_mikktspace(const ImporterMesh& mesh);

/// Generate one tangent and handedness value for every triangle corner without modifying the mesh.
/// Parallel output is Mikk-compatible and numerically equivalent, but not necessarily bit-identical.
/// Position, normal, and texture-coordinate values must be finite.
FALCOR_API std::vector<float4> generate_corner_tangent_space(const ImporterMesh& mesh, Generator generator);

/// Generate corner tangents with the selected implementation and assemble the required indexed vertex variants.
FALCOR_API void generate_tangent_space(ImporterMesh& mesh, Generator generator);

} // namespace falcor::mikkt_detail
#endif
