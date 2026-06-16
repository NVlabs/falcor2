// SPDX-License-Identifier: Apache-2.0
//
// Shared microfacet helpers ported from OSL/BSDL. The core GGX distribution
// uses Falcor's shared policy while bounded reflection helpers remain local.
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/microfacet_tools_impl.h
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/tools.h (reflect/refract)

implementing mx139;

import falcor2.render;

namespace mx139 {

// BSDL: tools.h:301-329 refract
// Not a direct wrapper around Slang builtin refract(I, N, eta): callers pass
// the BSDL incoming/outgoing convention used by these MX139 microfacet samplers.
// This helper orients the normal, inverts eta when needed, handles eta == 0 and
// total internal reflection, and normalizes the result.
public float3 mx_refract(float3 incident, float3 normal, float eta)
{
    if (eta == 0.0)
        return float3(0.0);

    float3 normal_oriented;
    float cosi = dot(incident, normal);
    float neta;
    if (cosi > 0.0) {
        neta = 1.0 / eta;
        normal_oriented = normal;
    } else {
        cosi = -cosi;
        neta = eta;
        normal_oriented = -normal;
    }

    float arg = 1.0 - (neta * neta * (1.0 - cosi * cosi));
    if (arg >= 0.0) {
        float dnp = sqrt(arg);
        float normal_scale = (neta * cosi) - dnp;
        return normalize(incident * (-neta) + normal_oriented * normal_scale);
    }
    return float3(0.0);
}

// BSDL: microfacet_tools_decl.h:13-47 GGXDist + microfacet_tools_impl.h:19-111
// (D, G1, G2_G1, sample, sample_for_refl, D_refl_D).
public struct MxGGXDist
{
    private AnisotropicGGXPolicy<10000, true> m_dist;

    public __init(float rough, float aniso, bool flip_aniso)
    {
        if (flip_aniso)
            aniso = -aniso;
        m_dist
            = AnisotropicGGXPolicy<10000, true>(float2(rough * rough * (1.0 + aniso), rough * rough * (1.0 - aniso)));
    }

    public __init(float2 alpha) { m_dist = AnisotropicGGXPolicy<10000, true>(alpha); }

    public __init() { m_dist = AnisotropicGGXPolicy<10000, true>(float2(1.0)); }

    public float D(float3 Hr) { return m_dist.eval_ndf(Hr); }

    public float G1(float3 w) { return m_dist.G1(w); }

    public float G2_G1(float3 wi, float3 wo) { return m_dist.G2(wi, wo) / m_dist.G1(wo); }

    public float3 sample(float3 wo, float randu, float randv)
    {
        const float2 alpha = m_dist.alpha;
        const float ax = alpha.x;
        const float ay = alpha.y;
        const float3 V = normalize(float3(ax * wo.x, ay * wo.y, wo.z));
        const float3 T1 = V.z < 0.9999 ? normalize(float3(V.y, -V.x, 0.0)) : float3(1.0, 0.0, 0.0);
        const float3 T2 = cross(T1, V);
        const float2 p = square_to_uniform_disk_concentric(float2(randu, randv));
        const float s = 0.5 * (1.0 + V.z);
        const float p2o = s * p.y + (1.0 - s) * sqrt(1.0 - p.x * p.x);
        const float p3 = sqrt(max(1.0 - p.x * p.x - p2o * p2o, 0.0));
        const float3 N = p.x * T1 + p2o * T2 + p3 * V;
        return normalize(float3(ax * N.x, ay * N.y, max(N.z, 0.0)));
    }

    public float3 sample_for_refl(float3 wo, float randu, float randv)
    {
        const float2 alpha = m_dist.alpha;
        const float ax = alpha.x;
        const float ay = alpha.y;
        float3 i_std = normalize(float3(wo.x * ax, wo.y * ay, wo.z));
        const float phi = 2.0 * PI * randu;
        const float a = clamp(min(ax, ay), 0.0, 1.0);
        const float s = 1.0 + sqrt(wo.x * wo.x + wo.y * wo.y);
        const float a2 = a * a;
        const float s2 = s * s;
        const float k = (1.0 - a2) * s2 / (s2 + a2 * wo.z * wo.z);
        const float b = k * i_std.z;
        const float z = (1.0 - randv) * (1.0 + b) - b;
        const float sin_theta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
        const float3 o_std = float3(sin_theta * cos(phi), sin_theta * sin(phi), z);
        const float3 m_std = i_std + o_std;
        return normalize(float3(m_std.x * ax, m_std.y * ay, m_std.z));
    }

    public float D_refl_D(float3 wo, float3 m)
    {
        const float2 alpha = m_dist.alpha;
        const float ax = alpha.x;
        const float ay = alpha.y;
        const float2 ai = float2(wo.x * ax, wo.y * ay);
        const float len2 = dot(ai, ai);
        const float t = sqrt(len2 + wo.z * wo.z);
        const float a = clamp(min(ax, ay), 0.0, 1.0);
        const float s = 1.0 + sqrt(wo.x * wo.x + wo.y * wo.y);
        const float a2 = a * a;
        const float s2 = s * s;
        const float k = (1.0 - a2) * s2 / (s2 + a2 * wo.z * wo.z);
        return 2.0 * wo.z / (k * wo.z + t);
    }
};

public float2 microfacet_clamped_roughness_xy(float2 roughness_xy)
{
    return float2(clamp(roughness_xy.x, BSDL_EPSILON, 2.0), clamp(roughness_xy.y, BSDL_EPSILON, 2.0));
}

public float microfacet_roughness(float2 roughness_xy)
{
    const float2 r = microfacet_clamped_roughness_xy(roughness_xy);
    const float ax = max(r.x, r.y);
    const float ay = min(r.x, r.y);
    const float b = ay / ax;
    const float aniso = (1.0 - b) / (1.0 + b);
    return sqrt(ax / (1.0 + aniso));
}

public float microfacet_aniso(float2 roughness_xy)
{
    const float2 r = microfacet_clamped_roughness_xy(roughness_xy);
    const float ax = max(r.x, r.y);
    const float ay = min(r.x, r.y);
    const float b = ay / ax;
    return (1.0 - b) / (1.0 + b);
}

public MxGGXDist make_ggx_dist(float2 roughness_xy)
{
    return MxGGXDist(microfacet_clamped_roughness_xy(roughness_xy));
}

public float mini_microfacet_get_cosine(int i)
{
    const float x = float(i) * (1.0 / float(MINI_MICROFACET_GGX_NC - 1));
    return max(x * x, 1e-6);
}

public float tabulated_energy_interpolate_mini_microfacet(float roughness, int i)
{
    float rf = roughness * float(MINI_MICROFACET_GGX_NR - 1);
    int ra = int(rf);
    int rb = min(ra + 1, MINI_MICROFACET_GGX_NR - 1);
    rf -= float(ra);
    uint ia = uint(ra * MINI_MICROFACET_GGX_NC + i);
    uint ib = uint(rb * MINI_MICROFACET_GGX_NC + i);
    return lerp(mini_microfacet_ggx_energy(ia), mini_microfacet_ggx_energy(ib), rf);
}

// BSDL: microfacet_tools_impl.h:178-210 TabulatedEnergyCurve::Emiss_eval (SPI Nf=1 path).
public float tabulated_energy_emiss_eval_mini_microfacet(float roughness, float c)
{
    float cos0 = mini_microfacet_get_cosine(0);
    if (c <= cos0)
        return tabulated_energy_interpolate_mini_microfacet(roughness, 0);
    for (int i = 1; i < MINI_MICROFACET_GGX_NC; i++) {
        const float cos1 = mini_microfacet_get_cosine(i);
        if (c < cos1) {
            const float q = (c - cos0) / (cos1 - cos0);
            return lerp(
                tabulated_energy_interpolate_mini_microfacet(roughness, i - 1),
                tabulated_energy_interpolate_mini_microfacet(roughness, i),
                q
            );
        }
        cos0 = cos1;
    }
    return tabulated_energy_interpolate_mini_microfacet(roughness, MINI_MICROFACET_GGX_NC - 1);
}

public float dielectric_get_cosine(int i)
{
    return max(float(i) * (1.0 / float(DIELECTRIC_NC - 1)), 1e-6);
}

public float
tabulated_energy_interpolate_dielectric(StructuredBuffer<float> table, float roughness, float fresnel_index, int i)
{
    float rf = roughness * float(DIELECTRIC_NR - 1);
    int ra = int(rf);
    int rb = min(ra + 1, DIELECTRIC_NR - 1);
    rf -= float(ra);

    float ff = fresnel_index * float(DIELECTRIC_NF - 1);
    int fa = int(ff);
    int fb = min(fa + 1, DIELECTRIC_NF - 1);
    ff -= float(fa);

    uint iaa = uint((fa * DIELECTRIC_NR + ra) * DIELECTRIC_NC + i);
    uint iab = uint((fa * DIELECTRIC_NR + rb) * DIELECTRIC_NC + i);
    uint iba = uint((fb * DIELECTRIC_NR + ra) * DIELECTRIC_NC + i);
    uint ibb = uint((fb * DIELECTRIC_NR + rb) * DIELECTRIC_NC + i);
    return lerp(lerp(table[iaa], table[iab], rf), lerp(table[iba], table[ibb], rf), ff);
}

// BSDL: microfacet_tools_impl.h:178-210 TabulatedEnergyCurve::Emiss_eval (MTX Nf=32 path).
public float
tabulated_energy_emiss_eval_dielectric(StructuredBuffer<float> table, float roughness, float fresnel_index, float c)
{
    float cos0 = dielectric_get_cosine(0);
    if (c <= cos0)
        return tabulated_energy_interpolate_dielectric(table, roughness, fresnel_index, 0);
    for (int i = 1; i < DIELECTRIC_NC; i++) {
        const float cos1 = dielectric_get_cosine(i);
        if (c < cos1) {
            const float q = (c - cos0) / (cos1 - cos0);
            return lerp(
                tabulated_energy_interpolate_dielectric(table, roughness, fresnel_index, i - 1),
                tabulated_energy_interpolate_dielectric(table, roughness, fresnel_index, i),
                q
            );
        }
        cos0 = cos1;
    }
    return tabulated_energy_interpolate_dielectric(table, roughness, fresnel_index, DIELECTRIC_NC - 1);
}

// BSDL: microfacet_tools_impl.h:342-372 eval_turquin_microms_reflection.
// BSDL's "microms" abbreviation means microfacet multiple scattering. Mx139
// keeps this as a generic GGX reflection lobe helper: callers provide the
// pre-evaluated Fresnel term and lobe scale, so the selected compensation
// remains visible at the BSDF call site.
public MxBsdfSample eval_ggx_reflection_lobe(MxGGXDist dist, float3 fresnel, float3 lobe_scale, float3 wi, float3 wo)
{
    const float cos_NO = wi.z;
    const float cos_NI = wo.z;
    if (cos_NI <= 0.0 || cos_NO <= 0.0)
        return MxBsdfSample();

    const float3 m = normalize(wi + wo);
    const float cos_MO = dot(m, wi);
    const float D = dist.D(m);
    const float D_refl_D = dist.D_refl_D(wi, m);
    const float D_refl = D_refl_D * D;
    const float G1 = dist.G1(wi);
    const float pdf = D_refl / (4.0 * cos_NO);
    const float out_value = dist.G2_G1(wo, wi) * G1 / D_refl_D;
    const float3 weight = out_value * fresnel * lobe_scale;
    return MxBsdfSample(wo, weight, pdf, -1.0);
}

// BSDL: microfacet_tools_impl.h:374-391 sample_turquin_microms_reflection.
// Samples the GGX reflection lobe and evaluates it through
// eval_ggx_reflection_lobe. Sampling owns the policy call because the
// microfacet normal and Fresnel term are discovered here; direct eval keeps the
// lobe scale explicit so compensation decisions stay visible in BSDF code.
public MxBsdfSample
sample_ggx_reflection_lobe<TFresnel : IMxFresnelPolicy, TCompensation : IMxMicrofacetCompensationPolicy>(
    MxGGXDist dist,
    TFresnel fresnel,
    float roughness,
    float3 wi,
    float3 rnd
)
{
    if (wi.z <= 0.0)
        return MxBsdfSample();

    float3 m = dist.sample_for_refl(wi, rnd.x, rnd.y);
    float3 wo = reflect(-wi, m);
    if (wo.z <= 0.0)
        return MxBsdfSample();
    const float3 F = fresnel.eval(dot(m, wi));
    return eval_ggx_reflection_lobe(dist, F, TCompensation::reflection_scale(roughness, wi.z, F), wi, wo);
}

} // namespace mx139
