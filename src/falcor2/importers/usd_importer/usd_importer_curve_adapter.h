// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "usd_importer_material.h"
#include "falcor2/importers/importer_types.h"
#include <optional>
#include <variant>

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usdGeom/basisCurves.h>
END_DISABLE_USD_WARNINGS

namespace falcor::usd_importer {
using CurveGeometry = std::variant<ImporterCurve, ImporterMesh>;

/// Read a curve prim once, selecting round curves or ribbon meshes from the first normal sample.
/// Motion reads points, widths and normals on one fixed-topology schedule. UVs remain static.
/// Returns no value for skipped round geometry; invalid oriented data produces a diagnostic.
std::optional<CurveGeometry> convert_usd_curves(
    const pxr::UsdGeomBasisCurves& curves,
    const ResolveMaterial& resolve_material,
    pxr::UsdTimeCode viewing_time,
    const ImportOptions& options
);
} // namespace falcor::usd_importer
