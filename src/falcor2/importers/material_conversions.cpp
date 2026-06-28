// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_conversions.h"

#include "falcor2/importers/usd_importer/usd_importer_material.h"
#include "falcor2/importers/usd_importer/usd_importer_mdl.h"
#include "falcor2/importers/usd_importer/usd_importer_materialx.h"
#include "falcor2/importers/usd_importer/usdshade_to_mtlx.h"

namespace falcor {

std::optional<Properties> convert_material(
    const ImporterMaterial& material,
    std::initializer_list<std::string_view> sources_list,
    std::initializer_list<std::string_view> targets_list
)
{
    for (auto& source : sources_list) {
        for (auto& target : targets_list) {
            if (source == "UsdPreviewSurface") {
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
