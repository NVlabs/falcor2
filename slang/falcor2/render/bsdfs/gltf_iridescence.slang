// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

namespace gltf_iridescence {

// Belcour/Barla Fourier-domain sensitivity fit, as used by the Khronos sample
// renderer (source/Renderer/shaders/iridescence.glsl, Apache-2.0).
[Differentiable]
float3 sensitivity(float optical_path_nm, float3 shift)
{
    float phase = 2.0 * float.PI * optical_path_nm * 1.0e-9;
    float3 variance = float3(4.3278e9, 9.3046e9, 6.6121e9);
    float3 xyz = float3(5.4856e-13, 4.4201e-13, 5.2481e-13) * sqrt(2.0 * float.PI* variance)
        * cos(float3(1.6810e6, 1.7953e6, 2.2084e6) * phase + shift) * exp(-phase * phase * variance);
    xyz.x += 9.7470e-14 * sqrt(2.0 * float.PI * 4.5282e9) * cos(2.2399e6 * phase + shift.x)
        * exp(-4.5282e9 * phase * phase);
    xyz /= 1.0685e-7;
    return mul(
        float3x3(3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108, 0.0415560, 0.0556434, -0.2040259, 1.0572252),
        xyz
    );
}

[Differentiable]
public float3 fresnel(float cosine, float film_ior, float thickness_nm, float3 base_f0)
{
    cosine = saturate(abs(cosine));
    float blend = saturate(thickness_nm / 0.03);
    blend = blend * blend * (3.0 - 2.0 * blend);
    float eta = lerp(1.0, max(film_ior, 1.0), blend);
    float cos_film = sqrt(max(0.0, 1.0 - (1.0 - cosine * cosine) / (eta * eta)));
    float r0 = (eta - 1.0) / (eta + 1.0);
    float r12 = eval_fresnel_schlick(float3(r0 * r0), float3(1.0), cosine).x;
    float t = 1.0 - r12;
    float3 root_f0 = sqrt(clamp(base_f0, 0.0, 0.9999));
    float3 base_ior = (1.0 + root_f0) / (1.0 - root_f0);
    float3 ratio = (base_ior - eta) / (base_ior + eta);
    float3 r23 = eval_fresnel_schlick(ratio * ratio, float3(1.0), cos_film);
    // Equal media have no second interface; Schlick's grazing term otherwise
    // leaves spurious interference whose phase depends on roundoff at equality.
    r23 = select(abs(ratio) < 1.0e-6, float3(0.0), r23);
    float3 phase = float.PI + select(base_ior < eta, float3(float.PI), float3(0.0));
    float optical_path = 2.0 * eta * max(thickness_nm, 0.0) * cos_film;
    float3 product = clamp(r12 * r23, 0.0, 0.9999);
    float3 rs = t * t * r23 / (1.0 - product);
    float3 result = r12 + rs;
    float3 amplitude = rs - t;
    for (int order = 1; order <= 2; ++order) {
        amplitude *= sqrt(product);
        result += 2.0 * amplitude * sensitivity(float(order) * optical_path, float(order) * phase);
    }
    // The RGB spectral approximation can leave the representable gamut.
    return max(result, 0.0);
}
}
