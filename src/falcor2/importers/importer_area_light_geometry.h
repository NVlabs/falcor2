// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/fwd.h"
#include "falcor2/importers/importer.h"

namespace falcor::detail {

/// Replace selected analytic area lights in an imported scene with emissive triangle geometry.
void convert_area_lights_to_geometry(ImporterScene& scene, const ImportOptions& options);

} // namespace falcor::detail
