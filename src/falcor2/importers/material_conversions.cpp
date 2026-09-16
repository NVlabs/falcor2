// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_conversions.h"

#include "falcor2/core/error.h"
#include "falcor2/importers/usd_importer/usd_importer_falcor_material.h"
#include "falcor2/importers/usd_importer/usd_importer_material.h"
#include "falcor2/importers/usd_importer/usd_importer_mdl.h"
#include "falcor2/importers/usd_importer/usd_importer_materialx.h"
#include "falcor2/importers/usd_importer/usdshade_to_mtlx.h"

#include <string>

namespace falcor {

namespace importer_material {

namespace {

std::string texture_srgb_property_name(std::string_view path_property_name)
{
    constexpr std::string_view path_suffix = "_path";
    FALCOR_CHECK(
        path_property_name.ends_with(path_suffix),
        "Texture path property '{}' must end in '_path'",
        path_property_name
    );

    std::string result(path_property_name.substr(0, path_property_name.size() - path_suffix.size()));
    result.append("_srgb");
    return result;
}

} // namespace

void set_texture_path(
    Properties& properties,
    std::string_view path_property_name,
    const std::filesystem::path& path,
    bool srgb
)
{
    properties.set(path_property_name, path);
    const std::string srgb_property_name = texture_srgb_property_name(path_property_name);
    if (srgb)
        properties.set(srgb_property_name, true);
    else
        properties.remove_property(srgb_property_name);
}

bool is_texture_srgb(const Properties& properties, std::string_view path_property_name)
{
    return properties.get<bool>(texture_srgb_property_name(path_property_name), false);
}

} // namespace importer_material

std::optional<Properties> convert_material(
    const ImporterMaterial& material,
    std::initializer_list<std::string_view> sources_list,
    std::initializer_list<std::string_view> targets_list
)
{
    for (auto& source : sources_list) {
        for (auto& target : targets_list) {
            if (source == "Falcor") {
                if (target == "StandardMaterial") {
                    if (auto result = falcor_to_standardmaterial(material))
                        return result;
                }
                if (target == "UnlitMaterial") {
                    if (auto result = falcor_to_unlitmaterial(material))
                        return result;
                }
                if (target == "OpenPBRMaterial") {
                    if (auto result = falcor_to_openpbrmaterial(material))
                        return result;
                }
            } else if (source == "UsdPreviewSurface") {
                if (target == "StandardMaterial") {
                    if (auto result = usdpreviewsurface_to_standardmaterial(material))
                        return result;
                }
                if (target == "MaterialXMaterial") {
                    if (auto result = usdpreviewsurface_to_mtlx(material))
                        return result;
                }
            } else if (source == "MDL") {
                if (target == "MDLMaterial") {
                    if (auto result = usd_to_mdl(material))
                        return result;
                }
            } else if (source == "MaterialX") {
                if (target == "MaterialXMaterial") {
                    if (auto result = usdshade_to_mtlx(material))
                        return result;
                    if (auto result = usd_to_mtlx(material))
                        return result;
                }
            }
        }
    }

    return {};
}

} // namespace falcor
