// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "usd_importer_macros.h"

#include "falcor2/core/properties.h"
#include "falcor2/core/types.h"


namespace falcor {

struct ImporterMaterial;

std::optional<Properties> usd_to_mdl(const ImporterMaterial& importer_material);

} // namespace falcor
