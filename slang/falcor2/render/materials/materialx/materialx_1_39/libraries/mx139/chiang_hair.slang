// SPDX-License-Identifier: Apache-2.0
//
// MaterialX Chiang hair BSDF.
//
// The eval/sample/pdf core follows Falcor1's HairChiang16 implementation,
// which is itself adapted from PBRT's Chiang et al. 2016 hair BSDF. The public
// parameters follow MaterialX 1.39 / OSL: per-lobe tints, per-lobe longitudinal
// variance and azimuthal logistic scale, IOR, cuticle angle, and absorption.
//
// MaterialX 1.39 only provides raster-style direct and indirect closures for
// GLSL, while OSL delegates to a renderer builtin. The MX139 layerable methods
// below are therefore best-effort for closure_tree layering: sampling and PDFs are
// high-confidence PBRT/Falcor1 behavior, while directional_albedo/layer_opacity
// are intentionally conservative approximations until hair participates in
// real layered material transport.

implementing mx139;

namespace mx139 {

static const int HAIR_P_MAX = 3;
static const float HAIR_SQRT_PI_OVER_8 = 0.626657069;
static const float HAIR_MIN_ROUGHNESS = 0.001;
static const float HAIR_MIN_IOR = 1.001;
static const float HAIR_MIN_COS_THETA = 1e-5;

float hair_sqr(float x)
{
    return x * x;
}

float hair_safe_sqrt(float x)
{
    return sqrt(max(x, 0.0));
}

float hair_safe_asin(float x)
{
    return asin(clamp(x, -1.0, 1.0));
}

float hair_wrap_pi(float phi)
{
    phi = fmod(phi, float.TWO_PI);
    if (phi > float.PI)
        phi -= float.TWO_PI;
    if (phi < -float.PI)
        phi += float.TWO_PI;
    return phi;
}

float hair_I0(float x)
{
    float value = 1.0;
    float x2i = 1.0;
    float ifact = 1.0;
    uint i4 = 1;

    [unroll]
    for (uint i = 1; i < 10; ++i) {
        ifact *= float(i);
        x2i *= x * x;
        i4 *= 4;
        value += x2i / (ifact * ifact * float(i4));
    }
    return value;
}

float hair_log_I0(float x)
{
    if (x > 12.0)
        return x + 0.5 * (-log(float.TWO_PI) + log(1.0 / x) + 0.125 / x);
    return log(hair_I0(x));
}

float hair_logistic(float x, float s)
{
    x = abs(x);
    float tmp = exp(-x / s);
    return tmp / (s * hair_sqr(1.0 + tmp));
}

float hair_logistic_cdf(float x, float s)
{
    return 1.0 / (1.0 + exp(-x / s));
}

float hair_trimmed_logistic(float x, float s, float a, float b)
{
    return hair_logistic(x, s) / max(hair_logistic_cdf(b, s) - hair_logistic_cdf(a, s), float.MIN);
}

float hair_sample_trimmed_logistic(float u, float s, float a, float b)
{
    float k = hair_logistic_cdf(b, s) - hair_logistic_cdf(a, s);
    float x = -s * log(1.0 / (u * k + hair_logistic_cdf(a, s)) - 1.0);
    return clamp(x, a, b);
}

float hair_phi(int p, float gamma_o, float gamma_t)
{
    float fp = float(p);
    return 2.0 * fp * gamma_t - 2.0 * gamma_o + fp * float.PI;
}

float hair_Np(float phi, int p, float s, float gamma_o, float gamma_t)
{
    if (p >= HAIR_P_MAX)
        return float.INV_TWO_PI;

    float dphi = hair_wrap_pi(phi - hair_phi(p, gamma_o, gamma_t));
    return hair_trimmed_logistic(dphi, s, -float.PI, float.PI);
}

float hair_Mp(float cos_theta_i, float cos_theta_o, float sin_theta_i, float sin_theta_o, float v)
{
    v = max(v, HAIR_MIN_ROUGHNESS);
    float a = cos_theta_i * cos_theta_o / v;
    float b = sin_theta_i * sin_theta_o / v;
    if (v <= 0.1)
        return exp(hair_log_I0(a) - b - 1.0 / v + 0.6931 + log(0.5 / v));
    return exp(-b) * hair_I0(a) / (sinh(1.0 / v) * 2.0 * v);
}

void hair_shift_view_theta(
    float alpha,
    int p,
    float sin_theta_o,
    float cos_theta_o,
    out float sin_theta_op,
    out float cos_theta_op
)
{
    // PBRT/Falcor1 shift the fixed view direction. This is equivalent to the
    // MaterialX 1.39 GLSL direct term, which shifts the sampled light direction
    // by the opposite lobe angle before evaluating Mp.
    if (p == 0) {
        float sa = sin(2.0 * alpha);
        float ca = cos(2.0 * alpha);
        sin_theta_op = sin_theta_o * ca - cos_theta_o * sa;
        cos_theta_op = cos_theta_o * ca + sin_theta_o * sa;
    } else if (p == 1) {
        float sa = sin(alpha);
        float ca = cos(alpha);
        sin_theta_op = sin_theta_o * ca + cos_theta_o * sa;
        cos_theta_op = cos_theta_o * ca - sin_theta_o * sa;
    } else if (p == 2) {
        float sa = sin(4.0 * alpha);
        float ca = cos(4.0 * alpha);
        sin_theta_op = sin_theta_o * ca + cos_theta_o * sa;
        cos_theta_op = cos_theta_o * ca - sin_theta_o * sa;
    } else {
        sin_theta_op = sin_theta_o;
        cos_theta_op = cos_theta_o;
    }
    cos_theta_op = abs(cos_theta_op);
}

float hair_view_h(float3 wi)
{
    float yz_len = length(wi.yz);
    if (yz_len <= HAIR_MIN_COS_THETA)
        return 0.0;
    return clamp(-wi.y / yz_len, -1.0, 1.0);
}

float hair_fresnel(float ior, float cos_theta)
{
    return tf_fresnel_dielectric(clamp(cos_theta, 0.0, 1.0), ior);
}

public struct MxChiangHairBSDF : ILayerableBSDF
{
    public float3 tint_R = {};
    public float3 tint_TT = {};
    public float3 tint_TRT = {};
    public float ior = {};
    public float2 roughness_R = {};
    public float2 roughness_TT = {};
    public float2 roughness_TRT = {};
    public float cuticle_angle = {};
    public float3 absorption_coefficient = {};

    public __init() { }

    public __init(
        float3 tint_R,
        float3 tint_TT,
        float3 tint_TRT,
        float ior,
        float2 roughness_R,
        float2 roughness_TT,
        float2 roughness_TRT,
        float cuticle_angle,
        float3 absorption_coefficient
    )
    {
        this.tint_R = max(tint_R, float3(0.0));
        this.tint_TT = max(tint_TT, float3(0.0));
        this.tint_TRT = max(tint_TRT, float3(0.0));
        this.ior = max(ior, HAIR_MIN_IOR);
        this.roughness_R = clamp(roughness_R, float2(HAIR_MIN_ROUGHNESS), float2(1.0));
        this.roughness_TT = clamp(roughness_TT, float2(HAIR_MIN_ROUGHNESS), float2(1.0));
        this.roughness_TRT = clamp(roughness_TRT, float2(HAIR_MIN_ROUGHNESS), float2(1.0));
        this.cuticle_angle = clamp(cuticle_angle, 0.0, 1.0);
        this.absorption_coefficient = max(absorption_coefficient, float3(0.0));
    }

    // ILayerableBSDF

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_value(wi, wo);
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
        wo = float3(0.0);
        pdf = 0.0;
        weight = float3(0.0);
        lobe_types = LobeTypes::none;


        float sin_theta_o = wi.x;
        float cos_theta_o = hair_safe_sqrt(1.0 - sin_theta_o * sin_theta_o);
        if (cos_theta_o <= HAIR_MIN_COS_THETA)
            return false;

        float4 u = next_float4(sg);
        float ap_pdf[HAIR_P_MAX + 1];
        compute_ap_pdf(wi, ap_pdf);

        int p = 0;
        float u_select = u.x;
        bool selection_done = false;
        [unroll]
        for (int i = 0; i < HAIR_P_MAX; ++i) {
            if (!selection_done && u_select >= ap_pdf[i]) {
                u_select -= ap_pdf[i];
                p = i + 1;
            } else {
                selection_done = true;
            }
        }

        float alpha = hair_alpha();
        float sin_theta_op;
        float cos_theta_op;
        hair_shift_view_theta(alpha, p, sin_theta_o, cos_theta_o, sin_theta_op, cos_theta_op);

        float vp = lobe_v(p);
        float u_m = max(u.z, 1e-5);
        float cos_theta = 1.0 + vp * log(u_m + (1.0 - u_m) * exp(-2.0 / vp));
        float sin_theta = hair_safe_sqrt(1.0 - cos_theta * cos_theta);
        float cos_phi = cos(u.w * float.TWO_PI);
        float sin_theta_i = -cos_theta * sin_theta_op + sin_theta * cos_phi * cos_theta_op;
        float cos_theta_i = hair_safe_sqrt(1.0 - sin_theta_i * sin_theta_i);

        float phi_o = atan2(wi.z, wi.y);
        float h = hair_view_h(wi);
        float gamma_o = hair_safe_asin(h);
        float eta_p = hair_safe_sqrt(ior * ior - sin_theta_o * sin_theta_o) / max(cos_theta_o, float.MIN);
        float gamma_t = hair_safe_asin(clamp(h / max(eta_p, float.MIN), -1.0, 1.0));

        float dphi = p < HAIR_P_MAX
            ? hair_phi(p, gamma_o, gamma_t) + hair_sample_trimmed_logistic(u.y, lobe_s(p), -float.PI, float.PI)
            : u.y * float.TWO_PI;
        float phi_i = phi_o + dphi;
        wo = float3(sin_theta_i, cos_theta_i * cos(phi_i), cos_theta_i * sin(phi_i));

        pdf = eval_pdf(wi, wo, bc);
        if (pdf <= 0.0 || isnan(pdf) || isinf(pdf)) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }

        weight = eval_value(wi, wo) / pdf;
        if (any(isnan(weight)) || any(isinf(weight)) || any(weight < float3(0.0))) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }

        lobe_types = LobeTypes::glossy_curve;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        float sin_theta_o = wi.x;
        float cos_theta_o = hair_safe_sqrt(1.0 - sin_theta_o * sin_theta_o);
        if (cos_theta_o <= HAIR_MIN_COS_THETA)
            return 0.0;

        float sin_theta_i = wo.x;
        float cos_theta_i = hair_safe_sqrt(1.0 - sin_theta_i * sin_theta_i);
        float phi = atan2(wo.z, wo.y) - atan2(wi.z, wi.y);

        float h = hair_view_h(wi);
        float gamma_o = hair_safe_asin(h);
        float eta_p = hair_safe_sqrt(ior * ior - sin_theta_o * sin_theta_o) / max(cos_theta_o, float.MIN);
        float gamma_t = hair_safe_asin(clamp(h / max(eta_p, float.MIN), -1.0, 1.0));

        float ap_pdf[HAIR_P_MAX + 1];
        compute_ap_pdf(wi, ap_pdf);

        float pdf = 0.0;
        float alpha = hair_alpha();
        [unroll]
        for (int p = 0; p < HAIR_P_MAX; ++p) {
            float sin_theta_op;
            float cos_theta_op;
            hair_shift_view_theta(alpha, p, sin_theta_o, cos_theta_o, sin_theta_op, cos_theta_op);
            pdf += ap_pdf[p] * hair_Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, lobe_v(p))
                * hair_Np(phi, p, lobe_s(p), gamma_o, gamma_t);
        }
        pdf += ap_pdf[HAIR_P_MAX] * hair_Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, lobe_v(HAIR_P_MAX))
            * float.INV_TWO_PI;

        return (isnan(pdf) || isinf(pdf)) ? 0.0 : pdf;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float3 reflection = saturate(directional_albedo(wi));
        return AlbedoContributions(reflection, float3(1.0) - reflection, float3(0.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation((roughness_R + roughness_TT + roughness_TRT) / 3.0, float3(0.0));
    }

    public float3 directional_albedo(const float3 wi)
    {
        float3 ap0;
        float3 ap1;
        float3 ap2;
        float3 ap3;
        attenuation(wi, ap0, ap1, ap2, ap3);
        return saturate(ap0 + ap1 + ap2 + ap3);
    }

    public float3 layer_opacity(const float3 wi) { return float3(1.0); }

    public property float3 ior_as_reflectance
    {
        get {
            float sqrt_f0 = (ior - 1.0) / (ior + 1.0);
            return float3(sqrt_f0 * sqrt_f0);
        }
    }

    // Implementation

    private float hair_alpha() { return cuticle_angle * float.PI - float.PI_OVER_TWO; }

    private float lobe_v(int p)
    {
        if (p == 0)
            return max(roughness_R.x, HAIR_MIN_ROUGHNESS);
        if (p == 1)
            return max(roughness_TT.x, HAIR_MIN_ROUGHNESS);
        return max(roughness_TRT.x, HAIR_MIN_ROUGHNESS);
    }

    private float lobe_s(int p)
    {
        float s = p == 0 ? roughness_R.y : (p == 1 ? roughness_TT.y : roughness_TRT.y);
        return max(s * HAIR_SQRT_PI_OVER_8, HAIR_MIN_ROUGHNESS);
    }

    private float3 eval_value(const float3 wi, const float3 wo)
    {
        float sin_theta_o = wi.x;
        float cos_theta_o = hair_safe_sqrt(1.0 - sin_theta_o * sin_theta_o);
        if (cos_theta_o <= HAIR_MIN_COS_THETA)
            return float3(0.0);

        float sin_theta_i = wo.x;
        float cos_theta_i = hair_safe_sqrt(1.0 - sin_theta_i * sin_theta_i);
        float phi = atan2(wo.z, wo.y) - atan2(wi.z, wi.y);

        float h = hair_view_h(wi);
        float gamma_o = hair_safe_asin(h);
        float eta_p = hair_safe_sqrt(ior * ior - sin_theta_o * sin_theta_o) / max(cos_theta_o, float.MIN);
        float gamma_t = hair_safe_asin(clamp(h / max(eta_p, float.MIN), -1.0, 1.0));

        float3 ap[HAIR_P_MAX + 1];
        attenuation(wi, ap[0], ap[1], ap[2], ap[3]);

        float3 value = float3(0.0);
        float alpha = hair_alpha();
        [unroll]
        for (int p = 0; p < HAIR_P_MAX; ++p) {
            float sin_theta_op;
            float cos_theta_op;
            hair_shift_view_theta(alpha, p, sin_theta_o, cos_theta_o, sin_theta_op, cos_theta_op);
            value += ap[p] * hair_Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, lobe_v(p))
                * hair_Np(phi, p, lobe_s(p), gamma_o, gamma_t);
        }
        value += ap[HAIR_P_MAX] * hair_Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, lobe_v(HAIR_P_MAX))
            * float.INV_TWO_PI;

        return (any(isnan(value)) || any(isinf(value))) ? float3(0.0) : max(value, float3(0.0));
    }

    private void attenuation(const float3 wi, out float3 ap0, out float3 ap1, out float3 ap2, out float3 ap3)
    {
        float sin_theta_o = wi.x;
        float cos_theta_o = max(hair_safe_sqrt(1.0 - sin_theta_o * sin_theta_o), HAIR_MIN_COS_THETA);
        float sin_theta_t = sin_theta_o / ior;
        float cos_theta_t = max(hair_safe_sqrt(1.0 - sin_theta_t * sin_theta_t), HAIR_MIN_COS_THETA);

        float h = hair_view_h(wi);
        float cos_gamma_o = hair_safe_sqrt(1.0 - h * h);
        float eta_p = hair_safe_sqrt(ior * ior - sin_theta_o * sin_theta_o) / cos_theta_o;
        float sin_gamma_t = clamp(h / max(eta_p, float.MIN), -1.0, 1.0);
        float cos_gamma_t = hair_safe_sqrt(1.0 - sin_gamma_t * sin_gamma_t);
        float3 T = exp(-absorption_coefficient * (2.0 * cos_gamma_t / cos_theta_t));

        float f = hair_fresnel(ior, cos_theta_o * cos_gamma_o);
        float one_minus_f = 1.0 - f;
        float3 Tf = T * f;
        float3 ap1_raw = T * (one_minus_f * one_minus_f);
        float3 ap2_raw = ap1_raw * Tf;
        float3 ap3_raw = ap2_raw * Tf / max(float3(1.0) - Tf, float3(BSDL_EPSILON));

        ap0 = float3(f) * tint_R;
        ap1 = ap1_raw * tint_TT;
        ap2 = ap2_raw * tint_TRT;
        ap3 = ap3_raw * tint_TRT;
    }

    private void compute_ap_pdf(const float3 wi, out float ap_pdf[HAIR_P_MAX + 1])
    {
        float3 ap[HAIR_P_MAX + 1];
        attenuation(wi, ap[0], ap[1], ap[2], ap[3]);

        float sum_y = 0.0;
        [unroll]
        for (int p = 0; p <= HAIR_P_MAX; ++p) {
            ap_pdf[p] = luminance(ap[p]);
            sum_y += ap_pdf[p];
        }

        if (sum_y <= 0.0) {
            [unroll]
            for (int p = 0; p <= HAIR_P_MAX; ++p)
                ap_pdf[p] = 1.0 / float(HAIR_P_MAX + 1);
            return;
        }

        float inv_sum_y = 1.0 / sum_y;
        [unroll]
        for (int p = 0; p <= HAIR_P_MAX; ++p)
            ap_pdf[p] *= inv_sum_y;
    }
}

public MxChiangHairBSDF create_mx_chiang_hair(
    float3 tint_R,
    float3 tint_TT,
    float3 tint_TRT,
    float ior,
    float2 roughness_R,
    float2 roughness_TT,
    float2 roughness_TRT,
    float cuticle_angle,
    float3 absorption_coefficient,
    float3 normal,
    float3 curve_direction
)
{
    return MxChiangHairBSDF(
        tint_R,
        tint_TT,
        tint_TRT,
        ior,
        roughness_R,
        roughness_TT,
        roughness_TRT,
        cuticle_angle,
        absorption_coefficient
    );
}

} // namespace mx139
