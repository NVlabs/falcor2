// SPDX-License-Identifier: Apache-2.0

void mx_roughness_anisotropy(float roughness, float anisotropy, out float2 result)
{
    float roughness_sqr = clamp(roughness * roughness, M_FLOAT_EPS, 1.0);
    if (anisotropy > 0.0) {
        float aspect = sqrt(1.0 - clamp(anisotropy, 0.0, 0.98));
        result.x = min(roughness_sqr / aspect, 1.0);
        result.y = roughness_sqr * aspect;
    } else {
        result.x = roughness_sqr;
        result.y = roughness_sqr;
    }
}

void mx_roughness_dual(float2 roughness, out float2 result)
{
    result.x = clamp(roughness.x * roughness.x, M_FLOAT_EPS, 1.0);
    result.y = roughness.y < 0.0 ? result.x : clamp(roughness.y * roughness.y, M_FLOAT_EPS, 1.0);
}

void mx_artistic_ior(float3 reflectivity, float3 edge_color, out float3 ior, out float3 extinction)
{
    float3 r = clamp(reflectivity, 0.0, 0.99);
    float3 r_sqrt = sqrt(r);
    float3 n_min = (1.0 - r) / (1.0 + r);
    float3 n_max = (1.0 + r_sqrt) / (1.0 - r_sqrt);
    ior = lerp(n_max, n_min, edge_color);

    float3 np1 = ior + 1.0;
    float3 nm1 = ior - 1.0;
    float3 k2 = (np1 * np1 * r - nm1 * nm1) / (1.0 - r);
    k2 = max(k2, 0.0);
    extinction = sqrt(k2);
}

void mx_deon_hair_absorption_from_melanin(
    float melanin_concentration,
    float melanin_redness,
    float3 eumelanin_color,
    float3 pheomelanin_color,
    out float3 absorption
)
{
    float melanin = -log(max(1.0 - melanin_concentration, 0.0001));
    float eumelanin = melanin * (1.0 - melanin_redness);
    float pheomelanin = melanin * melanin_redness;
    absorption = max(
        eumelanin * -log(max(eumelanin_color, float3(0.0001)))
            + pheomelanin * -log(max(pheomelanin_color, float3(0.0001))),
        float3(0.0)
    );
}

void mx_chiang_hair_absorption_from_color(float3 color, float azimuthal_roughness, out float3 absorption)
{
    float beta_n = clamp(azimuthal_roughness, 0.001, 1.0);
    float beta_n2 = beta_n * beta_n;
    float beta_n4 = beta_n2 * beta_n2;
    float b_fac = 5.969 - 0.215 * beta_n + 2.532 * beta_n2 - 10.73 * beta_n2 * beta_n + 5.574 * beta_n4
        + 0.245 * beta_n4 * beta_n;
    float3 sigma = log(clamp(color, float3(0.001), float3(1.0))) / b_fac;
    absorption = sigma * sigma;
}

void mx_chiang_hair_roughness(
    float longitudinal,
    float azimuthal,
    float scale_TT,
    float scale_TRT,
    out float2 roughness_R,
    out float2 roughness_TT,
    out float2 roughness_TRT
)
{
    float lr = clamp(longitudinal, 0.001, 1.0);
    float ar = clamp(azimuthal, 0.001, 1.0);

    float v = 0.726 * lr + 0.812 * lr * lr + 3.7 * pow(lr, 20.0);
    v *= v;

    float s = 0.265 * ar + 1.194 * ar * ar + 5.372 * pow(ar, 22.0);

    roughness_R = float2(v, s);
    roughness_TT = float2(v * scale_TT * scale_TT, s);
    roughness_TRT = float2(v * scale_TRT * scale_TRT, s);
}
