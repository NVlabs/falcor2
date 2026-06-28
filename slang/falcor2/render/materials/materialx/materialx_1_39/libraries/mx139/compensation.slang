// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Energy-compensation policies for Mx139 microfacet BSDFs.

implementing mx139;

namespace mx139 {

public interface IMxMicrofacetCompensationPolicy
{
    public static float3 reflection_scale(float roughness, float cos_theta, float3 fresnel);
}

public interface IMxRefractiveCompensationPolicy : IMxMicrofacetCompensationPolicy
{
    public static float layer_energy(
        bool has_reflection,
        bool has_refraction,
        bool backfacing,
        float refraction_eta,
        float roughness,
        float cos_theta
    );
    public static float3 refractive_reflection_scale(float roughness, float cos_theta, float3 fresnel);
    public static float refraction_scale(
        bool has_reflection,
        bool has_refraction,
        bool backfacing,
        float refraction_eta,
        float roughness,
        float cos_theta
    );
}

public float bsdl_dielectric_energy_table_index_from_eta(float eta)
{
    const float x = (dielectric_base_ior(eta) - 1.001) * (1.0 / (5.0 - 1.001));
    return sqrt(x);
}

// MaterialX 1.39 mx_microfacet_specular.glsl analytic GGX directional albedo
// fit, named DIRECTIONAL_ALBEDO_ANALYTIC in the raster implementation.
public float3 ggx_directional_albedo_analytic(float n_dot_v, float alpha, float3 F0, float3 F90)
{
    float x = n_dot_v;
    float y = alpha;
    float x2 = x * x;
    float y2 = y * y;
    float4 r = float4(0.1003, 0.9345, 1.0, 1.0) + float4(-0.6303, -2.323, -1.765, 0.2281) * x
        + float4(9.748, 2.229, 8.263, 15.94) * y + float4(-2.038, -3.748, 11.53, -55.83) * x * y
        + float4(29.34, 1.424, 28.96, 13.08) * x2 + float4(-8.245, -0.7684, -7.507, 41.26) * y2
        + float4(-26.44, 1.436, -36.11, 54.9) * x2 * y + float4(19.99, 0.2913, 15.86, 300.2) * x * y2
        + float4(-5.448, 0.6286, 33.37, -285.1) * x2 * y2;
    float2 ab = clamp(r.xy / r.zw, 0.0, 1.0);
    return F0 * ab.x + F90 * ab.y;
}

public float3 turquin_analytic_reflection_scale(float n_dot_v, float alpha, float3 first_bounce_fresnel)
{
    float first_bounce_energy = ggx_directional_albedo_analytic(n_dot_v, alpha, float3(1.0), float3(1.0)).x;
    return float3(1.0) + first_bounce_fresnel * (1.0 - first_bounce_energy) / max(first_bounce_energy, 1e-6);
}

// Layering pass-through for dielectric policies that do not have BSDL
// tabulated layer-energy curves. Lobe compensation controls how the upper
// microfacet lobe is scaled, but a reflection-only dielectric top still needs
// to expose the base layer by the macro Fresnel transmission share. Returning
// zero here makes the top layer fully opaque, which is only correct for a
// perfect mirror, not for ordinary dielectric reflection.
public float dielectric_no_compensation_layer_energy(
    bool has_reflection,
    bool has_refraction,
    bool backfacing,
    float refraction_eta,
    float roughness,
    float cos_theta
)
{
    if (has_reflection && !has_refraction) {
        const float F = dielectric_fresnel_eval(refraction_eta, abs(cos_theta));
        return 1.0 - F;
    }
    return 0.0;
}

// Turquin 2019, "Practical multiple scattering compensation for microfacet
// models" (https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf).
// This policy uses the OSL/BSDL TabulatedEnergyCurve<MiniMicrofacetGGX>
// missing-energy LUT and matches the BSDL helper historically named
// eval_turquin_microms_reflection:
// scale = 1 + F * E_miss / max(0.01, 1 - E_miss).
public struct MxTurquinLutMicrofacetCompensation : IMxMicrofacetCompensationPolicy
{
    [ForceInline]
    public static float3 reflection_scale(float roughness, float cos_theta, float3 fresnel)
    {
        const float E_ms = tabulated_energy_emiss_eval_mini_microfacet(roughness, cos_theta);
        return float3(1.0) + fresnel * (E_ms / max(0.01, 1.0 - E_ms));
    }
};

// Turquin 2019 lobe scaling with the MaterialX 1.39 analytic GGX directional
// albedo fit in mx_microfacet_specular.glsl:
// mx_ggx_dir_albedo_analytic plus mx_ggx_energy_compensation.
// MaterialX names this directional-albedo method DIRECTIONAL_ALBEDO_ANALYTIC.
public struct MxTurquinAnalyticMicrofacetCompensation : IMxMicrofacetCompensationPolicy
{
    [ForceInline]
    public static float3 reflection_scale(float roughness, float cos_theta, float3 fresnel)
    {
        return turquin_analytic_reflection_scale(abs(cos_theta), roughness, fresnel);
    }
};

// Explicit no-compensation policy for MaterialX 1.38/PBRT-style parity
// experiments. This keeps lobe scales at one, but still models the macro
// Fresnel pass-through that reflection-only dielectric layers need in order to
// expose their base layer. Do not use a blanket zero layer_energy() here: that
// makes reflection-only dielectric tops fully opaque.
public struct MxNoCompensationMicrofacetCompensation : IMxMicrofacetCompensationPolicy
{
    [ForceInline]
    public static float3 reflection_scale(float roughness, float cos_theta, float3 fresnel) { return float3(1.0); }
};

public struct MxNoCompensationDielectricCompensation : IMxRefractiveCompensationPolicy
{
    [ForceInline]
    public static float3 reflection_scale(float roughness, float cos_theta, float3 fresnel) { return float3(1.0); }

    [ForceInline]
    public static float layer_energy(
        bool has_reflection,
        bool has_refraction,
        bool backfacing,
        float refraction_eta,
        float roughness,
        float cos_theta
    )
    {
        return dielectric_no_compensation_layer_energy(
            has_reflection,
            has_refraction,
            backfacing,
            refraction_eta,
            roughness,
            cos_theta
        );
    }

    [ForceInline]
    public static float3 refractive_reflection_scale(float roughness, float cos_theta, float3 fresnel)
    {
        return float3(1.0);
    }

    [ForceInline]
    public static float refraction_scale(
        bool has_reflection,
        bool has_refraction,
        bool backfacing,
        float refraction_eta,
        float roughness,
        float cos_theta
    )
    {
        return 1.0;
    }
};

// Turquin LUT reflection compensation plus OSL/BSDL tabulated dielectric layer
// energies for reflection/refraction modes. The layer tables are BSDL
// TabulatedEnergyCurve<DielectricReflFront/DielectricBothFront/DielectricBothBack>.
public struct MxTurquinLutDielectricCompensation : IMxRefractiveCompensationPolicy
{
    [ForceInline]
    public static float dielectric_energy_table_index(float refraction_eta)
    {
        return bsdl_dielectric_energy_table_index_from_eta(refraction_eta);
    }

    [ForceInline]
    public static float3 reflection_scale(float roughness, float cos_theta, float3 fresnel)
    {
        return MxTurquinLutMicrofacetCompensation::reflection_scale(roughness, cos_theta, fresnel);
    }

    [ForceInline]
    public static float layer_energy(
        bool has_reflection,
        bool has_refraction,
        bool backfacing,
        float refraction_eta,
        float roughness,
        float cos_theta
    )
    {
        const float fresnel_index = dielectric_energy_table_index(refraction_eta);
        if (has_reflection && !has_refraction) {
            return tabulated_energy_emiss_eval_dielectric(
                lut_globals.dielectric_reflfront_energy,
                roughness,
                fresnel_index,
                cos_theta
            );
        }
        if (has_refraction) {
            if (backfacing) {
                return tabulated_energy_emiss_eval_dielectric(
                    lut_globals.dielectric_bothback_energy,
                    roughness,
                    fresnel_index,
                    cos_theta
                );
            }
            return tabulated_energy_emiss_eval_dielectric(
                lut_globals.dielectric_bothfront_energy,
                roughness,
                fresnel_index,
                cos_theta
            );
        }
        return 0.0;
    }

    [ForceInline]
    public static float3 refractive_reflection_scale(float roughness, float cos_theta, float3 fresnel)
    {
        return float3(1.0);
    }

    [ForceInline]
    public static float refraction_scale(
        bool has_reflection,
        bool has_refraction,
        bool backfacing,
        float refraction_eta,
        float roughness,
        float cos_theta
    )
    {
        if (!has_refraction)
            return 1.0;
        return 1.0
            / max(0.01,
                  1.0 - layer_energy(has_reflection, has_refraction, backfacing, refraction_eta, roughness, cos_theta));
    }
};

// Turquin 2019 lobe scaling with the MaterialX 1.39 analytic GGX directional
// albedo fit. Refraction remains uncompensated, matching the MaterialX 1.39
// raster BSDF helpers that apply mx_ggx_energy_compensation only to reflection.
public struct MxTurquinAnalyticDielectricCompensation : IMxRefractiveCompensationPolicy
{
    [ForceInline]
    public static float3 reflection_scale(float roughness, float cos_theta, float3 fresnel)
    {
        return MxTurquinAnalyticMicrofacetCompensation::reflection_scale(roughness, cos_theta, fresnel);
    }

    [ForceInline]
    public static float layer_energy(
        bool has_reflection,
        bool has_refraction,
        bool backfacing,
        float refraction_eta,
        float roughness,
        float cos_theta
    )
    {
        return dielectric_no_compensation_layer_energy(
            has_reflection,
            has_refraction,
            backfacing,
            refraction_eta,
            roughness,
            cos_theta
        );
    }

    [ForceInline]
    public static float3 refractive_reflection_scale(float roughness, float cos_theta, float3 fresnel)
    {
        return reflection_scale(roughness, cos_theta, fresnel);
    }

    [ForceInline]
    public static float refraction_scale(
        bool has_reflection,
        bool has_refraction,
        bool backfacing,
        float refraction_eta,
        float roughness,
        float cos_theta
    )
    {
        return 1.0;
    }
};

} // namespace mx139
