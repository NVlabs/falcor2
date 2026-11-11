// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mesh_tessellator.h"
#include "mesh_tessellator_internal.h"

namespace falcor {
namespace mesh_tessellator {

ImporterMesh tessellate(const TessellatorInputMesh& input)
{
    const int refinement_level = input.properties.get<int>("refinement_level");
    falcor::detail::validate_refinement_level(refinement_level);
    if (input.positions.empty() || input.positions[0].values.empty() || input.face_vertex_counts.empty()
        || input.face_vertex_indices.empty())
        return {};

    FALCOR_CHECK(
        input.normals.empty() || input.normals.size() == input.positions.size(),
        "Normal samples must match position samples"
    );

    if (input.subdivision_scheme == SubdivisionScheme::none || refinement_level == 0)
        return detail::triangulate_mesh(input);

    return detail::tessellate_subdivision_surface(input, refinement_level);
}

} // namespace mesh_tessellator
} // namespace falcor
