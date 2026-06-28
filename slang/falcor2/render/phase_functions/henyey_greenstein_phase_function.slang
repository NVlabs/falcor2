// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Henyey-Greenstein phase function.
/// Note: The incident and outgoing directions are assumed to be pointing away from the scattering position.
/// Note: This function reduces to isotropic phase function at g = 0 and has singularities at g = -1 and g = 1.
public struct HenyeyGreensteinPhaseFunction : IPhaseFunction
{
    float g;

    public __init(float g) { this.g = g; }

    public float3 eval(const float3 wi, const float3 wo) { return evalHenyeyGreenstein(dot(wi, wo), g); }

    public bool
    sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, inout S sg)
    {
        const float2 u = next_float2(sg);

        float cosTheta;
        if (abs(g) < 1e-3f) {
            cosTheta = 1.f - 2.f * u.x;
        } else {
            const float sqr = (1.f - g * g) / (1.f - g + 2.f * g * u.x);
            cosTheta = (1.f + g * g - sqr * sqr) / (2.f * g);
        }

        const float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        const float phi = float.TWO_PI * u.y;
        const float3 w = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

        const float3 dir = -wi;
        float3 T, B;
        branchless_ONB(dir, T, B);

        wo = normalize(w.x * T + w.y * B + w.z * dir);
        pdf = evalHenyeyGreenstein(-cosTheta, g);
        weight = 1.f;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo) { return evalHenyeyGreenstein(dot(wi, wo), g); }
};

/// Evaluates the anisotropic Henyey-Greenstein phase function.
/// Note: This function reduces to isotropic phase function at g = 0 and has singularities at g = -1 and g = 1.
/// Convention: both the incident direction and the outgoing direction point away from the scatter point, hence the "+"
/// @param cosTheta Cosine between incoming and scattered direction.
/// @param g Anisotropy parameter in (-1, 1), where positive values promote forward scattering.
float evalHenyeyGreenstein(const float cosTheta, const float g)
{
    const float denom = 1 + g * g + 2 * g * cosTheta;
    return float.INV_FOUR_PI * (1 - g * g) / (denom * sqrt(denom));
}
/// Evaluates the anisotropic Henyey-Greenstein phase function.
/// Note: This function reduces to isotropic phase function at g = 0 and has singularities at g = -1 and g = 1.
/// Convention: both the incident direction and the outgoing direction point away from the scatter point, hence the "+"
/// @param cosTheta Cosine between incoming and scattered direction.
/// @param g Anisotropy parameter in (-1, 1), where positive values promote forward scattering.
float3 evalHenyeyGreenstein(const float cosTheta, const float3 g)
{
    const float3 denom = 1 + g * g + 2 * g * cosTheta;
    return float.INV_FOUR_PI * (1 - g * g) / (denom * sqrt(denom));
}
