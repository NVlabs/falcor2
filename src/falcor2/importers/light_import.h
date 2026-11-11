// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "falcor2/render/component/light.h"
#include "falcor2/importers/importer_types.h"

namespace falcor {

/// Shared source-light conversion for importers. Reuses
/// a compatible component; otherwise removes previous and creates its replacement.
/// Set the entity transform first: point power depends on world-space scale.
Light* update_imported_light(Entity* entity, Light* previous, const ImporterLight& source);
} // namespace falcor
