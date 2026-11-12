// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "usd_importer_material.h"
#include "falcor2/importers/importer_types.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
END_DISABLE_USD_WARNINGS


namespace falcor {
namespace usd_importer {

/// Converts a USD mesh into an owning ImporterMesh before any borrowed USD storage expires.
ImporterMesh convert_usd_mesh(
    const pxr::UsdGeomMesh& usd_mesh,
    const ResolveMaterial& resolve_material,
    pxr::UsdTimeCode viewing_time,
    const ImportOptions& import_options,
    const pxr::UsdSkelSkinningQuery* skinning = nullptr
);

} // namespace usd_importer
} // namespace falcor
