// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Fresnel policies for Mx139 microfacet BSDFs.

implementing mx139;

namespace mx139 {

public float oriented_dielectric_eta(float eta_value, bool backside)
{
    if (backside)
        eta_value = 1.0 / eta_value;
    return eta_value >= 1.0 ? clamp(eta_value, 1.001, 5.0) : clamp(eta_value, 1.0 / 5.0, 1.0 / 1.001);
}

public float dielectric_base_ior(float eta)
{
    return clamp(eta < 1.0 ? 1.0 / eta : eta, 1.001, 5.0);
}

public float dielectric_tir_cos(float eta)
{
    return eta >= 1.0 ? 0.0 : sqrt(1.0 - eta * eta);
}

// BSDL: bsdf_dielectric_impl.h:21-64 DielectricFresnel
[ForceInline]
public float dielectric_fresnel_eval(float eta, float c)
{
    c = clamp(c, 0.0, 1.0);
    float g = (eta - 1.0) * (eta + 1.0) + c * c;
    if (g > 0.0) {
        g = sqrt(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
        return 0.5 * A * A * (1.0 + B * B);
    }
    return 1.0;
}

public float3 schlick_fresnel(float3 F0, float3 F90, float exponent, float c)
{
    return lerp(F0, F90, pow(1.0 - c, exponent));
}

public float3 schlick_color82_fresnel(float3 F0, float3 F82, float3 F90, float exponent, float c)
{
    const float cos_theta_max = 1.0 / 7.0;
    const float cos_theta_factor = 1.0 / (cos_theta_max * pow(1.0 - cos_theta_max, 6.0));
    float one_minus_max = 1.0 - cos_theta_max;
    float one_minus_c = 1.0 - c;
    float one_minus_c6 = one_minus_c * one_minus_c * one_minus_c * one_minus_c * one_minus_c * one_minus_c;
    float3 a = schlick_fresnel(F0, F90, exponent, cos_theta_max) * (float3(1.0) - F82) * cos_theta_factor;
    return schlick_fresnel(F0, F90, exponent, c) - a * c * one_minus_c6;
}

public interface IMxFresnelPolicy
{
    float3 eval(float cos_theta);
}

public interface IMxReflectiveFresnelPolicy : IMxFresnelPolicy
{
    float3 directional_albedo(float cos_theta, float roughness);
}

public interface IMxRefractiveFresnelPolicy : IMxFresnelPolicy
{
    float refraction_eta();
}

public struct MxConductorFresnel : IMxReflectiveFresnelPolicy
{
    public float3 IOR;
    public float3 extinction;

    public __init(float3 IOR, float3 extinction)
    {
        this.IOR = IOR;
        this.extinction = extinction;
    }

    public __init()
    {
        IOR = float3(1.0);
        extinction = float3(0.0);
    }

    // BSDL: bsdf_conductor_impl.h:19-42 ConductorFresnel::eval
    [ForceInline]
    public float3 eval(float cos_theta)
    {
        cos_theta = clamp(cos_theta, 0.0, 1.0);
        const float3 one = float3(1.0);
        const float3 cos2_theta = float3(cos_theta * cos_theta);
        const float3 sin2_theta = one - cos2_theta;
        const float3 n = IOR;
        const float3 k = extinction;
        const float3 n2 = n * n;
        const float3 k2 = k * k;
        const float3 t0 = n2 - k2 - sin2_theta;
        const float3 a2plusb2 = sqrt(t0 * t0 + 4.0 * n2 * k2);
        const float3 t1 = a2plusb2 + cos2_theta;
        const float3 a = sqrt(0.5 * (a2plusb2 + t0));
        const float3 t2 = (2.0 * cos_theta) * a;
        const float3 rs = (t1 - t2) / clamp(t1 + t2, float.MIN, BIG);
        const float3 t3 = cos2_theta * a2plusb2 + sin2_theta * sin2_theta;
        const float3 t4 = t2 * sin2_theta;
        const float3 rp = rs * (t3 - t4) / clamp(t3 + t4, float.MIN, BIG);
        return 0.5 * clamp(rp + rs, 0.0, 2.0);
    }

    // BSDL: bsdf_conductor_impl.h:62-76 ConductorFresnel::avg
    public float3 avg()
    {
        const float a = -0.32775145;
        const float b = 0.18346033;
        const float c = 0.61146583;
        const float d = -0.07785134;
        const float3 p = a + b * IOR + c * extinction + d * IOR * extinction;
        return clamp(p / (float3(1.0) + p), 0.0, 1.0);
    }

    [ForceInline]
    public float3 directional_albedo(float cos_theta, float roughness) { return avg(); }
};

public struct MxConductorAiryFresnel : IMxReflectiveFresnelPolicy
{
    public float3 IOR;
    public float3 extinction;
    public float thinfilm_thickness;
    public float thinfilm_ior;

    public __init(float3 IOR, float3 extinction, float thinfilm_thickness, float thinfilm_ior)
    {
        this.IOR = IOR;
        this.extinction = extinction;
        this.thinfilm_thickness = thinfilm_thickness;
        this.thinfilm_ior = thinfilm_ior;
    }

    public __init()
    {
        IOR = float3(1.0);
        extinction = float3(0.0);
        thinfilm_thickness = 0.0;
        thinfilm_ior = 1.5;
    }

    [ForceInline]
    public float3 eval(float cos_theta)
    {
        if (thinfilm_thickness <= 0.0)
            return MxConductorFresnel(IOR, extinction).eval(cos_theta);
        MxThinFilmConductorFresnelData fd
            = MxThinFilmConductorFresnelData(IOR, extinction, thinfilm_thickness, thinfilm_ior);
        return fd.compute_fresnel(clamp(cos_theta, 0.0, 1.0));
    }

    [ForceInline]
    public float3 directional_albedo(float cos_theta, float roughness)
    {
        if (thinfilm_thickness <= 0.0)
            return MxConductorFresnel(IOR, extinction).directional_albedo(cos_theta, roughness);
        MxThinFilmConductorFresnelData fd
            = MxThinFilmConductorFresnelData(IOR, extinction, thinfilm_thickness, thinfilm_ior);
        return fd.directional_albedo(clamp(cos_theta, 0.0, 1.0), roughness);
    }
};

public struct MxDielectricFresnel : IMxRefractiveFresnelPolicy
{
    public float eta;

    public __init(float eta_value, bool backside) { eta = oriented_dielectric_eta(eta_value, backside); }

    public __init() { eta = 1.5; }

    [ForceInline]
    public float eval_scalar(float c) { return dielectric_fresnel_eval(eta, c); }

    [ForceInline]
    public float3 eval(float c) { return float3(eval_scalar(c)); }

    [ForceInline]
    public float refraction_eta() { return eta; }
};

public struct MxDielectricAiryFresnel : IMxRefractiveFresnelPolicy
{
    public float eta;
    public float base_ior;
    public float tir_cos;
    public float thinfilm_thickness;
    public float thinfilm_ior;

    public __init(float eta_value, bool backside, float thinfilm_thickness, float thinfilm_ior)
    {
        eta = oriented_dielectric_eta(eta_value, backside);
        base_ior = dielectric_base_ior(eta);
        tir_cos = dielectric_tir_cos(eta);
        this.thinfilm_thickness = thinfilm_thickness;
        this.thinfilm_ior = thinfilm_ior;
    }

    public __init()
    {
        eta = 1.5;
        base_ior = 1.5;
        tir_cos = 0.0;
        thinfilm_thickness = 0.0;
        thinfilm_ior = 1.5;
    }

    [ForceInline]
    public float3 eval(float c)
    {
        c = clamp(c, 0.0, 1.0);
        if (c < tir_cos)
            return float3(1.0);
        if (thinfilm_thickness <= 0.0)
            return float3(dielectric_fresnel_eval(eta, c));

        MxThinFilmDielectricFresnelData fd
            = MxThinFilmDielectricFresnelData(base_ior, thinfilm_thickness, thinfilm_ior);
        return fd.compute_fresnel(c);
    }

    [ForceInline]
    public float refraction_eta() { return eta; }
};

// BSDL: bsdf_schlick_impl.h:16-35 SchlickFresnel (extends DielectricFresnel)
public struct MxSchlickFresnel : IMxRefractiveFresnelPolicy
{
    public float3 F0;
    public float3 F90;
    public float exponent;
    public float eta;
    public float tir_cos;

    public __init(float3 F0, float3 F90, float exponent, float eta_value, bool backfacing)
    {
        eta = oriented_dielectric_eta(eta_value, backfacing);
        this.F0 = clamp(F0, 0.0, 1.0);
        this.F90 = clamp(F90, 0.0, 1.0);
        this.exponent = exponent;
        tir_cos = dielectric_tir_cos(eta);
    }

    public __init()
    {
        F0 = float3(0.04);
        F90 = float3(1.0);
        exponent = 5.0;
        eta = 1.5;
        tir_cos = 0.0;
    }

    [ForceInline]
    public float3 eval(float c)
    {
        c = clamp(c, 0.0, 1.0);
        if (c < tir_cos)
            return float3(1.0);
        return schlick_fresnel(F0, F90, exponent, c);
    }

    [ForceInline]
    public float refraction_eta() { return eta; }
};

public struct MxSchlickColor82Fresnel : IMxRefractiveFresnelPolicy
{
    public float3 F0;
    public float3 F82;
    public float3 F90;
    public float exponent;
    public float eta;
    public float tir_cos;

    public __init(float3 F0, float3 F82, float3 F90, float exponent, float eta_value, bool backfacing)
    {
        eta = oriented_dielectric_eta(eta_value, backfacing);
        this.F0 = clamp(F0, 0.0, 1.0);
        this.F82 = max(F82, float3(0.0));
        this.F90 = clamp(F90, 0.0, 1.0);
        this.exponent = exponent;
        tir_cos = dielectric_tir_cos(eta);
    }

    public __init()
    {
        F0 = float3(0.04);
        F82 = float3(1.0);
        F90 = float3(1.0);
        exponent = 5.0;
        eta = 1.5;
        tir_cos = 0.0;
    }

    [ForceInline]
    public float3 eval(float c)
    {
        c = clamp(c, 0.0, 1.0);
        if (c < tir_cos)
            return float3(1.0);
        return schlick_color82_fresnel(F0, F82, F90, exponent, c);
    }

    [ForceInline]
    public float refraction_eta() { return eta; }
};

public struct MxSchlickAiryFresnel : IMxRefractiveFresnelPolicy
{
    public float3 F0;
    public float3 F82;
    public float3 F90;
    public float exponent;
    public float eta;
    public float tir_cos;
    public float thinfilm_thickness;
    public float thinfilm_ior;

    public __init(
        float3 F0,
        float3 F82,
        float3 F90,
        float exponent,
        float eta_value,
        bool backfacing,
        float thinfilm_thickness,
        float thinfilm_ior
    )
    {
        eta = oriented_dielectric_eta(eta_value, backfacing);
        this.F0 = clamp(F0, 0.0, 1.0);
        this.F82 = max(F82, float3(0.0));
        this.F90 = clamp(F90, 0.0, 1.0);
        this.exponent = exponent;
        tir_cos = dielectric_tir_cos(eta);
        this.thinfilm_thickness = thinfilm_thickness;
        this.thinfilm_ior = thinfilm_ior;
    }

    public __init()
    {
        F0 = float3(0.04);
        F82 = float3(1.0);
        F90 = float3(1.0);
        exponent = 5.0;
        eta = 1.5;
        tir_cos = 0.0;
        thinfilm_thickness = 0.0;
        thinfilm_ior = 1.5;
    }

    [ForceInline]
    public float3 eval(float c)
    {
        c = clamp(c, 0.0, 1.0);
        if (c < tir_cos)
            return float3(1.0);
        MxThinFilmSchlickFresnelData fd
            = MxThinFilmSchlickFresnelData(F0, F82, F90, exponent, thinfilm_thickness, thinfilm_ior);
        return fd.compute_fresnel(c);
    }

    [ForceInline]
    public float refraction_eta() { return eta; }
};

} // namespace mx139
