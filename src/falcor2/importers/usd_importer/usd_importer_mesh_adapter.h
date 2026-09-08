// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "usd_importer_macros.h"
#include "falcor2/importers/importer_types.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
END_DISABLE_USD_WARNINGS

#include <functional>
#include <string>
#include <string_view>

namespace falcor {
namespace usd_importer {

using ResolveMeshMaterial
    = std::function<std::string(const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view mesh_name)>;

/// Converts a USD mesh into an owning ImporterMesh before any borrowed USD storage expires.
ImporterMesh convert_usd_mesh(const pxr::UsdGeomMesh& usd_mesh, const ResolveMeshMaterial& resolve_material);

} // namespace usd_importer
} // namespace falcor
