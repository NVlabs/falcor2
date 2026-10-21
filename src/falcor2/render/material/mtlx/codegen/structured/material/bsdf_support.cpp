// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "bsdf_support.h"

#include <array>
#include <utility>

namespace falcor::mtlx::structured {

const BsdfInfo* find_bsdf_info(std::string_view implementation)
{
    static constexpr std::array<std::pair<std::string_view, BsdfInfo>, 12> BSDFS = {{
        {"IM_oren_nayar_diffuse_bsdf_genslangpt",
         {BsdfKind::oren_nayar_diffuse, "MxOrenNayarDiffuseBSDF", BsdfLobeFlags::diffuse_reflection}},
        {"IM_burley_diffuse_bsdf_genslangpt",
         {BsdfKind::burley_diffuse, "MxBurleyDiffuseBSDF", BsdfLobeFlags::diffuse_reflection}},
        {"IM_translucent_bsdf_genslangpt",
         {BsdfKind::translucent, "MxTranslucentBSDF", BsdfLobeFlags::diffuse_transmission, true}},
        {"IM_subsurface_bsdf_genslangpt",
         {BsdfKind::subsurface, "MxSubsurfaceDiffuseFallbackBSDF", BsdfLobeFlags::diffuse_reflection}},
        {"IM_beer_bsdf_genslangpt", {BsdfKind::beer, "MxBeerBSDF", BsdfLobeFlags::delta_transmission}},
        {"IM_scratchconductor_brdf_genslangpt",
         {BsdfKind::scratch_conductor, "MxScratchConductorBSDF", BsdfLobeFlags::glossy_reflection}},
        {"IM_dielectric_bsdf_genslangpt",
         {BsdfKind::dielectric, "", BsdfLobeFlags::glossy_reflection | BsdfLobeFlags::glossy_transmission, true}},
        {"IM_conductor_bsdf_genslangpt", {BsdfKind::conductor, "", BsdfLobeFlags::glossy_reflection}},
        {"IM_generalized_schlick_bsdf_genslangpt",
         {BsdfKind::generalized_schlick,
          "",
          BsdfLobeFlags::glossy_reflection | BsdfLobeFlags::glossy_transmission,
          true}},
        {"IM_sheen_bsdf_genslangpt", {BsdfKind::sheen, "", BsdfLobeFlags::glossy_reflection}},
        {"IM_chiang_hair_bsdf_genslangpt",
         {BsdfKind::chiang_hair, "MxChiangHairBSDF", BsdfLobeFlags::glossy_curve, false, true}},
        {"IM_surface_unlit_transparent_genslangpt",
         {BsdfKind::surface_unlit_transparent,
          "MxSurfaceUnlitTransparentBSDF",
          BsdfLobeFlags::delta_transmission,
          true}},
    }};
    for (const auto& [name, info] : BSDFS) {
        if (implementation == name)
            return &info;
    }
    return nullptr;
}

} // namespace falcor::mtlx::structured
