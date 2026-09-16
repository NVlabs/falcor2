// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/importer_types.h"

#include "falcor2/core/macros.h"

#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string_view>

namespace falcor {

inline const std::initializer_list<std::string_view> DEFAULT_MATERIAL_TARGETS = {
    "StandardMaterial",
    "UnlitMaterial",
    "OpenPBRMaterial",
    "MDLMaterial",
    "MaterialXMaterial",
};

inline const std::initializer_list<std::string_view> DEFAULT_MATERIAL_SOURCES = {
    "Falcor",
    "UsdPreviewSurface",
    "MDL",
    "MaterialX",
};

namespace importer_material {

/// Set an imported material texture path and add its sRGB sibling property when requested.
FALCOR_API void set_texture_path(
    Properties& properties,
    std::string_view path_property_name,
    const std::filesystem::path& path,
    bool srgb = true
);

/// Return whether the sRGB sibling property is set for an imported texture path property.
FALCOR_API bool is_texture_srgb(const Properties& properties, std::string_view path_property_name);

} // namespace importer_material

FALCOR_API std::optional<Properties> convert_material(
    const ImporterMaterial& material,
    std::initializer_list<std::string_view> sources_list = DEFAULT_MATERIAL_SOURCES,
    std::initializer_list<std::string_view> targets_list = DEFAULT_MATERIAL_TARGETS
);

} // namespace falcor
