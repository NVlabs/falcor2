// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "usd_importer_macros.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/timeCode.h>
END_DISABLE_USD_WARNINGS

namespace falcor::usd_importer {

/// Bake a numeric-time snapshot in place, using refinement zero on skinned meshes.
/// Requires an importer-owned stage with a private edit layer. Does nothing if no skinning is found.
/// Instanced skinning is unsupported.
void bake_skinning(const pxr::UsdStageRefPtr& stage, pxr::UsdTimeCode time);

/// Bake meshes at their position/normal sample times, without adding skeleton key times or freezing the scene.
/// Meshes without samples are baked at numeric time zero. Uses refinement zero on skinned meshes.
/// Requires an importer-owned stage with a private edit layer. Blend shapes and instanced skinning are unsupported.
void bake_skinning_animated(const pxr::UsdStageRefPtr& stage);

} // namespace falcor::usd_importer
