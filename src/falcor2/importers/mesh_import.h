// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/importers/importer_types.h"

namespace falcor {
class DynamicMeshGeometry;

/// Upload imported attributes and material slots, converting UVs to the target origin.
void update_imported_mesh(StaticMeshGeometry* geometry, const ImporterMesh& mesh, UVOrigin uv_origin);
void update_imported_mesh(DynamicMeshGeometry* geometry, const ImporterMesh& mesh, UVOrigin uv_origin);
} // namespace falcor
