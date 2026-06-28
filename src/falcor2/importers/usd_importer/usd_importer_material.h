// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "usd_importer_macros.h"

#include "falcor2/core/properties.h"
#include "falcor2/core/types.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/pxr.h>
#include <pxr/usd/usdShade/material.h>
END_DISABLE_USD_WARNINGS

#include <functional>

namespace falcor {

struct ImporterMaterial;
struct ImporterTexture;

namespace usd_importer {

/// NOTE: Material graph extraction in build_material is adapted in part from
/// OpenUSD's UsdImaging material graph utilities
/// (pxr/usdImaging/usdImaging/materialParamUtils.cpp).
/// Adds material with the given terminal_identifier (surface, volume, displacement) to the material.
void build_material(
    ImporterMaterial& material,
    const pxr::UsdPrim& usd_terminal,
    const std::string_view& terminal_identifier,
    const pxr::TfTokenVector& shaderSourceTypes,
    const pxr::TfTokenVector& renderContexts
);

} // namespace usd_importer

std::optional<Properties> usdpreviewsurface_to_flattened(
    const ImporterMaterial& importer_material,
    std::function<int(const ImporterTexture&)> add_texture_to_scene
);
std::optional<Properties> usdpreviewsurface_to_standardmaterial(const ImporterMaterial& importer_material);
std::optional<Properties> usdpreviewsurface_to_mtlx(const ImporterMaterial& importer_material);

} // namespace falcor
