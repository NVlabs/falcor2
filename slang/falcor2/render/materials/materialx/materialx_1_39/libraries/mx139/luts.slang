// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Runtime-indexed BSDL LUT payloads for mx139. Payloads are owned and
// uploaded by the Python/C++ harness; this module only defines dimensions and
// buffer access.
//
// Dimensions match BSDL exactly:
//   - mini_microfacet_ggx_energy: 16 x 16, BSDL microfacet_tools_decl.h:67-69 (Nc x Nr x Nf=1)
//   - dielectric_{reflfront, bothfront, bothback}: 32 x 16 x 16, BSDL
//     bsdf_dielectric_decl.h:34-36 (Nf x Nr x Nc = 32 x 16 x 16)
//   - zeltner_sheen_ltc_param: 32 x 32 x 3, BSDL bsdf_sheen_decl.h:155 (ltc_res=32)
//     storing three coefficients (a_inv, b_inv, r_coeff) per (roughness, cos_NO) cell.

implementing mx139;

public static constexpr int MINI_MICROFACET_GGX_NC = 16;
public static constexpr int MINI_MICROFACET_GGX_NR = 16;
public static constexpr int MINI_MICROFACET_GGX_NF = 1;
public static constexpr int DIELECTRIC_NC = 16;
public static constexpr int DIELECTRIC_NR = 16;
public static constexpr int DIELECTRIC_NF = 32;
public static constexpr int ZELTNER_SHEEN_LTC_RES = 32;

namespace mx139 {

public struct MxLutGlobals
{
    public StructuredBuffer<float> mini_microfacet_ggx_energy;
    public StructuredBuffer<float> dielectric_reflfront_energy;
    public StructuredBuffer<float> dielectric_bothfront_energy;
    public StructuredBuffer<float> dielectric_bothback_energy;
    public StructuredBuffer<float> zeltner_sheen_ltc_param;
};

} // namespace mx139

public mx139::MxLutGlobals lut_globals;

public float mini_microfacet_ggx_energy(uint index)
{
    return lut_globals.mini_microfacet_ggx_energy[index];
}

public float dielectric_reflfront_energy(uint index)
{
    return lut_globals.dielectric_reflfront_energy[index];
}

public float dielectric_bothfront_energy(uint index)
{
    return lut_globals.dielectric_bothfront_energy[index];
}

public float dielectric_bothback_energy(uint index)
{
    return lut_globals.dielectric_bothback_energy[index];
}

public float3 zeltner_sheen_ltc_param(uint row, uint col)
{
    uint index = (row * uint(ZELTNER_SHEEN_LTC_RES) + col) * 3;
    return float3(
        lut_globals.zeltner_sheen_ltc_param[index],
        lut_globals.zeltner_sheen_ltc_param[index + 1],
        lut_globals.zeltner_sheen_ltc_param[index + 2]
    );
}
