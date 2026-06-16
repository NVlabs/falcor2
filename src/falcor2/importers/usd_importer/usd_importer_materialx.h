// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "usd_importer_macros.h"

#include "falcor2/core/properties.h"
#include "falcor2/core/types.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
END_DISABLE_USD_WARNINGS

namespace falcor {
struct ImporterMaterial;

namespace usd_importer {

/// Expand Scope prims that reference .mtlx files into UsdShade materials and shaders.
/// Returns number of materials generated.
size_t process_mtlx_references(const pxr::UsdStageRefPtr& stage);

} // namespace usd_importer

std::optional<Properties> usd_to_mtlx(const ImporterMaterial& importer_material);

} // namespace falcor
