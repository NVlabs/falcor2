// SPDX-License-Identifier: Apache-2.0

#include "profile_policy.h"

namespace falcor {
namespace materialx {
namespace mx139 {

const ProfileDesc& profile_desc()
{
    static const ProfileDesc k_mx139 = {
        "MX139",
        "mx139",
        "Mx",
        "falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.mx139",
        "Mx",
        "MxStackData",
    };
    return k_mx139;
}

std::string fresnel_bsdf_type(const ProfileDesc& profile, const CodeGenDesc& desc, std::string_view stem)
{
    auto apply_compensation = [&](std::string type)
    {
        const CompensationMode compensation_mode = desc.compensation_mode;
        if (compensation_mode == CompensationMode::turquin_lut)
            return type;
        constexpr std::string_view suffix = "BSDF";
        if (type.size() < suffix.size() || type.compare(type.size() - suffix.size(), suffix.size(), suffix) != 0)
            return type;
        type.resize(type.size() - suffix.size());
        if (compensation_mode == CompensationMode::turquin_analytic)
            return type + "TurquinAnalyticCompensationBSDF";
        if (compensation_mode == CompensationMode::no_compensation)
            return type + "NoCompensationBSDF";
        return type + std::string(suffix);
    };

    if (stem == "Dielectric")
        return apply_compensation("MxDielectricAiryBSDF");
    if (stem == "Conductor")
        return apply_compensation("MxConductorAiryBSDF");
    if (stem == "GeneralizedSchlick")
        return apply_compensation("MxGeneralizedSchlickAiryBSDF");
    return apply_compensation(profile.bsdf_type(stem));
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
