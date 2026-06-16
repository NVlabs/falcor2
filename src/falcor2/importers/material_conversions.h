// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/importer_types.h"

#include "falcor2/core/macros.h"

#include <initializer_list>
#include <optional>
#include <string_view>

namespace falcor {

inline const std::initializer_list<std::string_view> DEFAULT_MATERIAL_TARGETS = {
    "StandardMaterial",
    "MDLMaterial",
    "MaterialXMaterial",
};

inline const std::initializer_list<std::string_view> DEFAULT_MATERIAL_SOURCES = {
    "UsdPreviewSurface",
    "MDL",
    "MaterialX",
};

FALCOR_API std::optional<Properties> convert_material(
    const ImporterMaterial& material,
    std::initializer_list<std::string_view> sources_list = DEFAULT_MATERIAL_SOURCES,
    std::initializer_list<std::string_view> targets_list = DEFAULT_MATERIAL_TARGETS
);

} // namespace falcor
