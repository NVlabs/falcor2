// SPDX-License-Identifier: Apache-2.0
//
// Thin-film Airy Fresnel helpers for Mx139 microfacet Fresnel policies.

implementing mx139;

namespace mx139 {

static const int TF_MODEL_DIELECTRIC = 0;
static const int TF_MODEL_CONDUCTOR = 1;
static const int TF_MODEL_SCHLICK = 2;
static const int TF_AIRY_FRESNEL_ITERATIONS = 2;

// Keep compatibility with the shared MaterialX 1.39 thin-film switch while
// making the Mx139 Fresnel implementation independent from that helper file.
#ifndef MX139_THIN_FILM_USE_OPENPBR_PT_ENABLED
#ifdef MX139_THIN_FILM_USE_OPENPBR_PT
#define MX139_THIN_FILM_USE_OPENPBR_PT_ENABLED MX139_THIN_FILM_USE_OPENPBR_PT
#else
#define MX139_THIN_FILM_USE_OPENPBR_PT_ENABLED 0
#endif
#endif

static const float TF_OPENPBR_MIN_DENOM = 1e-9;
static const float3 TF_OPENPBR_RGB_WAVELENGTHS_NM = float3(620.0, 540.0, 450.0);

float3 tf_f0_to_ior(float3 f0)
{
    float3 sqrt_f0 = sqrt(clamp(f0, 0.01, 0.99));
    return (float3(1.0) + sqrt_f0) / (float3(1.0) - sqrt_f0);
}

float tf_fresnel_dielectric(float cos_theta, float ior)
{
    float c = cos_theta;
    float g2 = ior * ior + c * c - 1.0;
    if (g2 < 0.0)
        return 1.0;

    float g = sqrt(g2);
    float g_minus_c = g - c;
    float g_plus_c = g + c;
    float numerator = (g_plus_c * c) - 1.0;
    float denominator = (g_minus_c * c) + 1.0;
    return 0.5 * (g_minus_c * g_minus_c) / (g_plus_c * g_plus_c)
        * (1.0 + (numerator * numerator) / (denominator * denominator));
}

float2 tf_fresnel_dielectric_polarized(float cos_theta, float ior)
{
    float cos_theta2 = clamp(cos_theta, 0.0, 1.0);
    cos_theta2 *= cos_theta2;
    float sin_theta2 = 1.0 - cos_theta2;

    float t0 = max(ior * ior - sin_theta2, 0.0);
    float t1 = t0 + cos_theta2;
    float t2 = 2.0 * sqrt(t0) * cos_theta;
    float rs = (t1 - t2) / (t1 + t2);

    float t3 = cos_theta2 * t0 + sin_theta2 * sin_theta2;
    float t4 = t2 * sin_theta2;
    float rp = rs * (t3 - t4) / (t3 + t4);

    return float2(rp, rs);
}

void tf_fresnel_conductor_polarized(float cos_theta, float3 n, float3 k, out float3 rp, out float3 rs)
{
    float cos_theta2 = clamp(cos_theta, 0.0, 1.0);
    cos_theta2 *= cos_theta2;
    float sin_theta2 = 1.0 - cos_theta2;
    float3 n2 = n * n;
    float3 k2 = k * k;

    float3 t0 = n2 - k2 - float3(sin_theta2);
    float3 a2_plus_b2 = sqrt(t0 * t0 + 4.0 * n2 * k2);
    float3 t1 = a2_plus_b2 + float3(cos_theta2);
    float3 a = sqrt(max(0.5 * (a2_plus_b2 + t0), 0.0));
    float3 t2 = 2.0 * a * cos_theta;
    rs = (t1 - t2) / (t1 + t2);

    float3 t3 = cos_theta2 * a2_plus_b2 + float3(sin_theta2 * sin_theta2);
    float3 t4 = t2 * sin_theta2;
    rp = rs * (t3 - t4) / (t3 + t4);
}

float3 tf_fresnel_conductor(float cos_theta, float3 n, float3 k)
{
    float3 rp;
    float3 rs;
    tf_fresnel_conductor_polarized(cos_theta, n, k, rp, rs);
    return 0.5 * (rp + rs);
}

void tf_fresnel_conductor_phase_polarized(
    float cos_theta,
    float eta1,
    float3 eta2,
    float3 kappa2,
    out float3 phi_p,
    out float3 phi_s
)
{
    float3 k2 = kappa2 / eta2;
    float3 sin_theta_sq = float3(1.0 - cos_theta * cos_theta);
    float3 A = eta2 * eta2 * (float3(1.0) - k2 * k2) - eta1 * eta1 * sin_theta_sq;
    float3 B = sqrt(A * A + (2.0 * eta2 * eta2 * k2) * (2.0 * eta2 * eta2 * k2));
    float3 U = sqrt((A + B) * 0.5);
    float3 V = max(float3(0.0), sqrt((B - A) * 0.5));

    phi_s = float3(
        atan2(2.0 * eta1 * V.x * cos_theta, U.x * U.x + V.x * V.x - (eta1 * cos_theta) * (eta1 * cos_theta)),
        atan2(2.0 * eta1 * V.y * cos_theta, U.y * U.y + V.y * V.y - (eta1 * cos_theta) * (eta1 * cos_theta)),
        atan2(2.0 * eta1 * V.z * cos_theta, U.z * U.z + V.z * V.z - (eta1 * cos_theta) * (eta1 * cos_theta))
    );

    float3 numer = 2.0 * eta1 * eta2 * eta2 * cos_theta * (2.0 * k2 * U - (float3(1.0) - k2 * k2) * V);
    float3 denom
        = (eta2 * eta2 * (float3(1.0) + k2 * k2) * cos_theta) * (eta2 * eta2 * (float3(1.0) + k2 * k2) * cos_theta)
        - eta1 * eta1 * (U * U + V * V);
    phi_p = float3(atan2(numer.x, denom.x), atan2(numer.y, denom.y), atan2(numer.z, denom.z));
}

float3 tf_eval_sensitivity(float opd, float3 shift)
{
    float phase = float.TWO_PI * opd;
    float3 val = float3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    float3 pos = float3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    float3 var = float3(4.3278e+09, 9.3046e+09, 6.6121e+09);

    float3 xyz = float3(
        val.x * sqrt(float.TWO_PI* var.x) * cos(pos.x * phase + shift.x) * exp(-var.x * phase * phase),
        val.y * sqrt(float.TWO_PI* var.y) * cos(pos.y * phase + shift.y) * exp(-var.y * phase * phase),
        val.z * sqrt(float.TWO_PI* var.z) * cos(pos.z * phase + shift.z) * exp(-var.z * phase * phase)
    );
    xyz.x += 9.7470e-14 * sqrt(float.TWO_PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x)
        * exp(-4.5282e+09 * phase * phase);
    return xyz / 1.0685e-7;
}

float3 tf_xyz_to_rgb(float3 xyz)
{
    return clamp(
        float3(
            2.3706743 * xyz.x - 0.9000405 * xyz.y - 0.4706338 * xyz.z,
            -0.5138850 * xyz.x + 1.4253036 * xyz.y + 0.0885814 * xyz.z,
            0.0052982 * xyz.x - 0.0146949 * xyz.y + 1.0093968 * xyz.z
        ),
        0.0,
        1.0
    );
}

float tf_safe_denom(float value)
{
    if (abs(value) >= TF_OPENPBR_MIN_DENOM)
        return value;
    return value >= 0.0 ? TF_OPENPBR_MIN_DENOM : -TF_OPENPBR_MIN_DENOM;
}

float2 tf_complex_mul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 tf_complex_div(float2 a, float2 b, float2 fallback)
{
    float denom = b.x * b.x + b.y * b.y;
    if (denom < TF_OPENPBR_MIN_DENOM * TF_OPENPBR_MIN_DENOM)
        return fallback;
    return float2((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}

float2 tf_complex_sqrt(float2 z)
{
    float m = sqrt(z.x * z.x + z.y * z.y);
    float imag_sign = copysign(1.0, z.y);
    return float2(sqrt(max(0.5 * (m + z.x), 0.0)), imag_sign * sqrt(max(0.5 * (m - z.x), 0.0)));
}

float2 tf_complex_exp_i(float angle)
{
    return float2(cos(angle), sin(angle));
}

float tf_complex_mag2(float2 z)
{
    return z.x * z.x + z.y * z.y;
}

float2 tf_openpbr_enforce_decaying_branch(float2 eta_t, float2 cos_theta_t)
{
    float2 eta_t_cos = tf_complex_mul(eta_t, cos_theta_t);
    return eta_t_cos.y < 0.0 ? -cos_theta_t : cos_theta_t;
}

float2 tf_openpbr_snell_cos_unified(float cos_theta_i, float eta_i, float2 eta_t)
{
    float sin_theta_i = sqrt(clamp(1.0 - cos_theta_i * cos_theta_i, 0.0, 1.0));
    float2 eta_i_over_eta_t = tf_complex_div(float2(eta_i, 0.0), eta_t, float2(1.0, 0.0));
    float2 sin_theta_t = sin_theta_i * eta_i_over_eta_t;
    float2 cos_theta_t_sq = float2(1.0, 0.0) - tf_complex_mul(sin_theta_t, sin_theta_t);
    return tf_openpbr_enforce_decaying_branch(eta_t, tf_complex_sqrt(cos_theta_t_sq));
}

void tf_openpbr_fresnel_unified_polarized_amplitude(
    float cos_theta_i,
    float eta_i,
    float2 eta_t,
    out float2 r_s,
    out float2 r_p,
    out float2 cos_theta_t
)
{
    cos_theta_t = tf_openpbr_snell_cos_unified(cos_theta_i, eta_i, eta_t);

    float2 eta_i_cos_theta_i = float2(eta_i * cos_theta_i, 0.0);
    float2 eta_t_cos_theta_t = tf_complex_mul(eta_t, cos_theta_t);
    float2 eta_t_cos_theta_i = cos_theta_i * eta_t;
    float2 eta_i_cos_theta_t = eta_i * cos_theta_t;

    r_s = tf_complex_div(
        eta_i_cos_theta_i - eta_t_cos_theta_t,
        eta_i_cos_theta_i + eta_t_cos_theta_t,
        float2(1.0, 0.0)
    );
    r_p = tf_complex_div(
        eta_t_cos_theta_i - eta_i_cos_theta_t,
        eta_t_cos_theta_i + eta_i_cos_theta_t,
        float2(1.0, 0.0)
    );
}

void tf_openpbr_fresnel_dielectric_polarized_amplitude(
    float cos_theta_i,
    float eta_i,
    float eta_t,
    out float r_s,
    out float r_p,
    out float t_s,
    out float t_p,
    out float cos_theta_t
)
{
    float sin_theta_i = sqrt(clamp(1.0 - cos_theta_i * cos_theta_i, 0.0, 1.0));
    float sin_theta_t = eta_i / eta_t * sin_theta_i;

    if (sin_theta_t >= 1.0) {
        r_s = 1.0;
        r_p = 1.0;
        t_s = 0.0;
        t_p = 0.0;
        cos_theta_t = 0.0;
        return;
    }

    cos_theta_t = sqrt(clamp(1.0 - sin_theta_t * sin_theta_t, 0.0, 1.0));

    float eta_i_cos_theta_i = eta_i * cos_theta_i;
    float eta_t_cos_theta_t = eta_t * cos_theta_t;
    float eta_t_cos_theta_i = eta_t * cos_theta_i;
    float eta_i_cos_theta_t = eta_i * cos_theta_t;

    float denom_s = tf_safe_denom(eta_i_cos_theta_i + eta_t_cos_theta_t);
    float denom_p = tf_safe_denom(eta_t_cos_theta_i + eta_i_cos_theta_t);
    r_s = (eta_i_cos_theta_i - eta_t_cos_theta_t) / denom_s;
    r_p = (eta_t_cos_theta_i - eta_i_cos_theta_t) / denom_p;
    t_s = (2.0 * eta_i_cos_theta_i) / denom_s;
    t_p = (2.0 * eta_i_cos_theta_i) / denom_p;
}

float tf_openpbr_airy_reflectance(float r12, float t12, float r21, float t21, float2 r23, float delta_phi)
{
    float2 phase = tf_complex_exp_i(delta_phi);
    float2 numerator = (t12 * t21) * tf_complex_mul(r23, phase);
    float2 denominator = float2(1.0, 0.0) - r21 * tf_complex_mul(r23, phase);
    float2 total = float2(r12, 0.0) + tf_complex_div(numerator, denominator, float2(0.0));
    return tf_complex_mag2(total);
}

float tf_openpbr_channel_reflectance(
    float cos_theta_i,
    float eta_exterior,
    float eta_film,
    float eta_base,
    float kappa_base,
    float thinfilm_thickness_nm,
    float wavelength_nm
)
{
    float r12_s;
    float r12_p;
    float t12_s;
    float t12_p;
    float cos_theta_t_film;
    tf_openpbr_fresnel_dielectric_polarized_amplitude(
        cos_theta_i,
        eta_exterior,
        eta_film,
        r12_s,
        r12_p,
        t12_s,
        t12_p,
        cos_theta_t_film
    );

    if (cos_theta_t_film <= 0.0)
        return 1.0;

    float r21_s = -r12_s;
    float r21_p = -r12_p;
    float safe_cos_theta_i = max(cos_theta_i, TF_OPENPBR_MIN_DENOM);
    float t21_scale = (eta_film * cos_theta_t_film) / (eta_exterior * safe_cos_theta_i);
    float t21_s = t12_s * t21_scale;
    float t21_p = t12_p * t21_scale;

    float2 r23_s;
    float2 r23_p;
    float2 cos_theta_t_base_unused;
    tf_openpbr_fresnel_unified_polarized_amplitude(
        cos_theta_t_film,
        eta_film,
        float2(eta_base, kappa_base),
        r23_s,
        r23_p,
        cos_theta_t_base_unused
    );

    float opd = 2.0 * eta_film * thinfilm_thickness_nm * cos_theta_t_film;
    float delta_phi = float.TWO_PI * opd / wavelength_nm;
    float reflectance_s = tf_openpbr_airy_reflectance(r12_s, t12_s, r21_s, t21_s, r23_s, delta_phi);
    float reflectance_p = tf_openpbr_airy_reflectance(r12_p, t12_p, r21_p, t21_p, r23_p, delta_phi);
    return 0.5 * (reflectance_s + reflectance_p);
}

float3 tf_fresnel_airy_openpbr_pt(
    float cos_theta,
    float thinfilm_thickness_nm,
    float thinfilm_ior,
    float3 eta_base,
    float3 kappa_base
)
{
    float cos_theta_i = clamp(cos_theta, 0.0, 1.0);
    float eta_exterior = 1.0;
    float eta_film = max(thinfilm_ior, eta_exterior);
    float3 eta_base_safe = max(eta_base, float3(TF_OPENPBR_MIN_DENOM));

    return clamp(
        float3(
            tf_openpbr_channel_reflectance(
                cos_theta_i,
                eta_exterior,
                eta_film,
                eta_base_safe.x,
                max(kappa_base.x, 0.0),
                thinfilm_thickness_nm,
                TF_OPENPBR_RGB_WAVELENGTHS_NM.x
            ),
            tf_openpbr_channel_reflectance(
                cos_theta_i,
                eta_exterior,
                eta_film,
                eta_base_safe.y,
                max(kappa_base.y, 0.0),
                thinfilm_thickness_nm,
                TF_OPENPBR_RGB_WAVELENGTHS_NM.y
            ),
            tf_openpbr_channel_reflectance(
                cos_theta_i,
                eta_exterior,
                eta_film,
                eta_base_safe.z,
                max(kappa_base.z, 0.0),
                thinfilm_thickness_nm,
                TF_OPENPBR_RGB_WAVELENGTHS_NM.z
            )
        ),
        0.0,
        1.0
    );
}

public struct MxThinFilmFresnelData<let MODEL : int>
{
    public bool airy;
    public float3 ior;
    public float3 extinction;
    public float3 f0;
    public float3 f82;
    public float3 f90;
    public float exponent;
    public float thinfilm_thickness;
    public float thinfilm_ior;

    public __init() { }

    public __init(float ior, float thinfilm_thickness, float thinfilm_ior)
    {
        this.airy = thinfilm_thickness > 0.0;
        this.ior = float3(ior);
        this.extinction = float3(0.0);
        this.f0 = float3(0.0);
        this.f82 = float3(0.0);
        this.f90 = float3(0.0);
        this.exponent = 0.0;
        this.thinfilm_thickness = thinfilm_thickness;
        this.thinfilm_ior = thinfilm_ior;
    }

    public __init(float3 ior, float3 extinction, float thinfilm_thickness, float thinfilm_ior)
    {
        this.airy = thinfilm_thickness > 0.0;
        this.ior = ior;
        this.extinction = extinction;
        this.f0 = float3(0.0);
        this.f82 = float3(0.0);
        this.f90 = float3(0.0);
        this.exponent = 0.0;
        this.thinfilm_thickness = thinfilm_thickness;
        this.thinfilm_ior = thinfilm_ior;
    }

    public __init(float3 f0, float3 f82, float3 f90, float exponent, float thinfilm_thickness, float thinfilm_ior)
    {
        this.airy = thinfilm_thickness > 0.0;
        this.ior = float3(0.0);
        this.extinction = float3(0.0);
        this.f0 = f0;
        this.f82 = f82;
        this.f90 = f90;
        this.exponent = exponent;
        this.thinfilm_thickness = thinfilm_thickness;
        this.thinfilm_ior = thinfilm_ior;
    }

    public float3 compute_fresnel(float cos_theta)
    {
        if (airy)
            return eval_fresnel_airy(cos_theta);
        if (MODEL == TF_MODEL_DIELECTRIC)
            return float3(tf_fresnel_dielectric(cos_theta, ior.x));
        if (MODEL == TF_MODEL_CONDUCTOR)
            return tf_fresnel_conductor(cos_theta, ior, extinction);
        return eval_fresnel_hoffman_schlick(cos_theta);
    }

    public float3 directional_albedo(float cos_theta, float alpha)
    {
        if (airy) {
            float3 mirror_dir_albedo = compute_fresnel(cos_theta);
            float3 F0 = eval_fresnel_airy(1.0);
            float3 rough_dir_albedo = ggx_directional_albedo_analytic(cos_theta, alpha, F0, float3(1.0));
            return lerp(mirror_dir_albedo, rough_dir_albedo, sqrt(alpha));
        }

        if (MODEL == TF_MODEL_DIELECTRIC) {
            float F0 = ((ior.x - 1.0) / (ior.x + 1.0));
            F0 *= F0;
            return ggx_directional_albedo_analytic(cos_theta, alpha, float3(F0), float3(1.0));
        }

        if (MODEL == TF_MODEL_CONDUCTOR) {
            float3 F0 = tf_fresnel_conductor(1.0, ior, extinction);
            return ggx_directional_albedo_analytic(cos_theta, alpha, F0, float3(1.0));
        }

        return ggx_directional_albedo_analytic(cos_theta, alpha, f0, f90);
    }

    float3 eval_fresnel_hoffman_schlick(float cos_theta)
    {
        const float cos_theta_max = 1.0 / 7.0;
        const float cos_theta_factor = 1.0 / (cos_theta_max * pow(1.0 - cos_theta_max, 6.0));

        float x = clamp(cos_theta, 0.0, 1.0);
        float3 a = lerp(f0, f90, pow(1.0 - cos_theta_max, exponent)) * (float3(1.0) - f82) * cos_theta_factor;
        float one_minus_x = 1.0 - x;
        float one_minus_x6 = one_minus_x * one_minus_x * one_minus_x * one_minus_x * one_minus_x * one_minus_x;
        return lerp(f0, f90, pow(one_minus_x, exponent)) - a * x * one_minus_x6;
    }

    float3 eval_fresnel_airy(float cos_theta)
    {
#if MX139_THIN_FILM_USE_OPENPBR_PT_ENABLED
        float3 eta3 = (MODEL == TF_MODEL_SCHLICK) ? tf_f0_to_ior(f0) : ior;
        float3 kappa3 = (MODEL == TF_MODEL_CONDUCTOR) ? extinction : float3(0.0);
        return tf_fresnel_airy_openpbr_pt(cos_theta, thinfilm_thickness, thinfilm_ior, eta3, kappa3);
#else
        return eval_fresnel_airy_materialx(cos_theta);
#endif
    }

    float3 eval_fresnel_airy_materialx(float cos_theta)
    {
        float eta1 = 1.0;
        float eta2 = max(thinfilm_ior, eta1);
        float3 eta3 = (MODEL == TF_MODEL_SCHLICK) ? tf_f0_to_ior(f0) : ior;
        float3 kappa3 = (MODEL == TF_MODEL_SCHLICK) ? float3(0.0) : extinction;
        float eta1_over_eta2 = eta1 / eta2;
        float cos_theta_t = sqrt(max(1.0 - (1.0 - cos_theta * cos_theta) * (eta1_over_eta2 * eta1_over_eta2), 0.0));

        float2 R12 = tf_fresnel_dielectric_polarized(cos_theta, eta2 / eta1);
        if (cos_theta_t <= 0.0)
            R12 = float2(1.0);
        float2 T121 = float2(1.0) - R12;

        float3 R23p;
        float3 R23s;
        if (MODEL == TF_MODEL_SCHLICK) {
            float3 f = eval_fresnel_hoffman_schlick(cos_theta_t);
            R23p = 0.5 * f;
            R23s = 0.5 * f;
        } else {
            tf_fresnel_conductor_polarized(cos_theta_t, eta3 / eta2, kappa3 / eta2, R23p, R23s);
        }

        float cos_B = rsqrt(1.0 + (eta2 / eta1) * (eta2 / eta1));
        float2 phi21 = float2(cos_theta < cos_B ? 0.0 : float.PI, float.PI);
        float3 phi23p;
        float3 phi23s;
        if (MODEL == TF_MODEL_SCHLICK) {
            phi23p = float3(
                eta3.x < eta2 ? float.PI : 0.0,
                eta3.y < eta2 ? float.PI : 0.0,
                eta3.z < eta2 ? float.PI : 0.0
            );
            phi23s = phi23p;
        } else {
            tf_fresnel_conductor_phase_polarized(cos_theta_t, eta2, eta3, kappa3, phi23p, phi23s);
        }

        float3 r123p = max(sqrt(R12.x * R23p), 0.0);
        float3 r123s = max(sqrt(R12.y * R23s), 0.0);

        float3 I = float3(0.0);
        float dist_meters = thinfilm_thickness * 1.0e-9;
        float opd = 2.0 * eta2 * cos_theta_t * dist_meters;

        float3 Rs = ((T121.x * T121.x) * R23p) / (float3(1.0) - R12.x * R23p);
        I += R12.x + Rs;

        float3 Cm = Rs - T121.x;
        for (int m = 1; m <= TF_AIRY_FRESNEL_ITERATIONS; ++m) {
            Cm *= r123p;
            float3 Sm = 2.0 * tf_eval_sensitivity(float(m) * opd, float(m) * (phi23p + float3(phi21.x)));
            I += Cm * Sm;
        }

        float3 Rp = ((T121.y * T121.y) * R23s) / (float3(1.0) - R12.y * R23s);
        I += R12.y + Rp;

        Cm = Rp - T121.y;
        for (int m = 1; m <= TF_AIRY_FRESNEL_ITERATIONS; ++m) {
            Cm *= r123s;
            float3 Sm = 2.0 * tf_eval_sensitivity(float(m) * opd, float(m) * (phi23s + float3(phi21.y)));
            I += Cm * Sm;
        }

        I *= 0.5;
        return tf_xyz_to_rgb(I);
    }
};

public typealias MxThinFilmDielectricFresnelData = MxThinFilmFresnelData<TF_MODEL_DIELECTRIC>;
public typealias MxThinFilmConductorFresnelData = MxThinFilmFresnelData<TF_MODEL_CONDUCTOR>;
public typealias MxThinFilmSchlickFresnelData = MxThinFilmFresnelData<TF_MODEL_SCHLICK>;

} // namespace mx139
