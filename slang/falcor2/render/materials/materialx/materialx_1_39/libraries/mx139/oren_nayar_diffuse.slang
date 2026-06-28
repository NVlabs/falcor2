// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Oren-Nayar diffuse BSDF ported from OSL/BSDL MTX.
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_oren_nayar_diffuse_impl.h

implementing mx139;

namespace mx139 {

static const float OREN_NAYAR_CONSTANT1_FON = 0.5 - 2.0 / (3.0 * float.PI);
static const float OREN_NAYAR_CONSTANT2_FON = 2.0 / 3.0 - 28.0 / (15.0 * float.PI);

public struct MxOrenNayarDiffuseBSDF : ILayerableBSDF
{
    public float3 albedo_value = {};
    public float roughness = {};
    public bool energy_compensation = {};

    public __init() { }

    public __init(float3 albedo_value, float roughness, bool energy_compensation)
    {
        this.albedo_value = albedo_value;
        this.roughness = clamp(roughness, 0.0, 1.0);
        this.energy_compensation = energy_compensation;
    }

    // ILayerableBSDF

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        MxBsdfSample r = eval_bsdf_sample(wi, wo);
        return r.pdf > 0.0 ? r.weight * r.pdf : float3(0.0);
    }

    public bool sample<S : ISampleGenerator>(
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 weight,
        out LobeTypes lobe_types,
        inout S sg,
        const BSDFContext bc
    )
    {
        MxBsdfSample r = sample_bsdf_sample(wi, next_float3(sg));
        if (r.pdf <= 0.0) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        wo = r.wo;
        pdf = r.pdf;
        weight = r.weight;
        lobe_types = LobeTypes::diffuse_reflection;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        return eval_bsdf_sample(wi, wo).pdf;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        const float3 reflection = saturate(albedo_value);
        return AlbedoContributions(reflection, float3(1.0) - reflection, float3(0.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(0.5, 0.5), float3(0.0));
    }

    public float3 directional_albedo(const float3 wi) { return albedo_value; }

    public float3 layer_opacity(const float3 wi) { return float3(1.0); }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }

    // Implementation

    // BSDL: bsdf_oren_nayar_diffuse_impl.h:27-40 E_FON_analytic
    private float E_FON_analytic(float mu)
    {
        const float sigma = roughness;
        const float AF = 1.0 / (1.0 + OREN_NAYAR_CONSTANT1_FON * sigma);
        const float BF = sigma * AF;
        const float Si = sqrt(max(0.0, 1.0 - mu * mu));
        const float G = Si * (acos(clamp(mu, -1.0, 1.0)) - Si * mu)
            + 2.0 * ((Si / max(mu, 1e-7)) * (1.0 - Si * Si * Si) - Si) * (1.0 / 3.0);
        return AF + (BF * float.INV_PI) * G;
    }

    // BSDL: bsdf_oren_nayar_diffuse_impl.h:42-97 eval_impl, translated to
    // Falcor wi/wo naming.
    private MxBsdfSample eval_bsdf_sample(const float3 wi, const float3 wo)
    {
        const float cos_NO = wi.z;
        const float cos_NI = wo.z;
        if (cos_NI <= 0.0 || cos_NO <= 0.0)
            return MxBsdfSample();

        const float cos_IO = clamp(dot(wo, wi), -1.0, 1.0);
        const float s = cos_IO - cos_NI * cos_NO;
        const float pdf = cos_NI * float.INV_PI;

        if (!energy_compensation) {
            const float s2 = roughness * roughness;
            const float A = 1.0 - 0.50 * s2 / (s2 + 0.33);
            const float B = 0.45 * s2 / (s2 + 0.09);
            const float stinv = s > 0.0 ? s / max(cos_NI, cos_NO) : 0.0;
            const float f_ss = A + B * stinv;
            return MxBsdfSample(wo, albedo_value * f_ss, pdf, 1.0);
        }

        const float AF = 1.0 / (1.0 + OREN_NAYAR_CONSTANT1_FON * roughness);
        const float stinv = s > 0.0 ? s / max(cos_NI, cos_NO) : s;
        const float f_ss = AF * (1.0 + roughness * stinv);
        const float EFo = E_FON_analytic(cos_NO);
        const float EFi = E_FON_analytic(cos_NI);
        const float avgEF = AF * (1.0 + OREN_NAYAR_CONSTANT2_FON * roughness);
        const float f_ms = max(1e-7, 1.0 - EFo) * max(1e-7, 1.0 - EFi) / max(1e-7, 1.0 - avgEF);
        float3 rho_ms = (albedo_value * albedo_value) * avgEF / (float3(1.0) - albedo_value * max(0.0, 1.0 - avgEF));
        return MxBsdfSample(wo, albedo_value * f_ss + rho_ms * f_ms, pdf, 1.0);
    }

    // BSDL: bsdf_oren_nayar_diffuse_impl.h:99-109 sample_impl
    private MxBsdfSample sample_bsdf_sample(const float3 wi, const float3 rnd)
    {
        if (wi.z <= 0.0)
            return MxBsdfSample();
        return eval_bsdf_sample(wi, square_to_cosine_hemisphere_concentric(rnd.xy));
    }
};

public MxOrenNayarDiffuseBSDF
create_mx_oren_nayar(float3 color, float roughness, float3 normal, bool energy_compensation)
{
    // Mx139 follows the authored MaterialX value and enables BSDL's
    // compensated Oren-Nayar path when requested.
    return MxOrenNayarDiffuseBSDF(color, roughness, energy_compensation);
}

} // namespace mx139
