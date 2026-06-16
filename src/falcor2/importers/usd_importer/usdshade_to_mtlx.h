// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/properties.h"

#include <optional>

namespace falcor {

struct ImporterMaterial;

// NOTE: This converter is adapted in part from OpenUSD's hdMtlx
// material network translation approach.

/// Convert a UsdShade-derived MaterialX node network (implementationSource=id)
/// into a generated .mtlx document reference consumable by MaterialXMaterial.
std::optional<Properties> usdshade_to_mtlx(const ImporterMaterial& importer_material);

} // namespace falcor
