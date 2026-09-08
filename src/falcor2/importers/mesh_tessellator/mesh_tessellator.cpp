// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mesh_tessellator.h"
#include "mesh_tessellator_internal.h"

namespace falcor {
namespace mesh_tessellator {

ImporterMesh tessellate(const TessellatorInputMesh& input)
{
    if (input.positions.values.empty() || input.face_vertex_counts.empty() || input.face_vertex_indices.empty())
        return {};

    if (input.subdivision_scheme == SubdivisionScheme::none || input.refinement_level == 0)
        return detail::triangulate_mesh(input);

    return detail::tessellate_subdivision_surface(input);
}

} // namespace mesh_tessellator
} // namespace falcor
