// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/properties.h"

#include <optional>

namespace falcor {

struct ImporterMaterial;

/// Convert a Falcor USD render-context terminal to StandardMaterial properties.
std::optional<Properties> falcor_to_standardmaterial(const ImporterMaterial& importer_material);

/// Convert a Falcor USD render-context terminal to UnlitMaterial properties.
std::optional<Properties> falcor_to_unlitmaterial(const ImporterMaterial& importer_material);

/// Convert a Falcor USD render-context terminal to OpenPBRMaterial properties.
std::optional<Properties> falcor_to_openpbrmaterial(const ImporterMaterial& importer_material);

} // namespace falcor
