// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

static constexpr float HENYEY_GREENSTEIN_ISOTROPIC_THRESHOLD = 1e-3;

/// Scalar Henyey-Greenstein phase function.
///
/// The mean cosine g controls anisotropy. Positive values produce forward scattering, negative values produce backward
/// scattering, and g = 0 reduces to isotropic scattering. The supported range is -1 < g < 1; the endpoints are
/// singular distributions.
public struct HenyeyGreensteinPhaseFunction : IPhaseFunction
{
    /// Mean cosine controlling the scattering anisotropy.
    float g;

    /// Constructs a Henyey-Greenstein phase function.
    /// @param g Mean cosine in the range (-1, 1).
    public __init(float g) { this.g = g; }

    public float3 eval(const float3 wi, const float3 wo) { return eval_henyey_greenstein(dot(wi, wo), g); }

    public bool sample<S : ISampleGenerator>(const float3 wi, inout S sg, out PhaseFunctionSample sample)
    {
        const float2 u = next_float2(sg);

        float cos_theta;
        if (abs(g) < HENYEY_GREENSTEIN_ISOTROPIC_THRESHOLD) {
            cos_theta = 1.0 - 2.0 * u.x;
        } else {
            const float sqr = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
            cos_theta = (1.0 + g * g - sqr * sqr) / (2.0 * g);
        }

        const float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        const float phi = float.TWO_PI * u.y;
        const float3 w = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

        const float3 dir = -wi;
        float3 t, b;
        branchless_ONB(dir, t, b);

        sample.wo_ws = normalize(w.x * t + w.y * b + w.z * dir);
        sample.pdf = eval_pdf(wi, sample.wo_ws);
        sample.weight = eval(wi, sample.wo_ws) / sample.pdf;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo)
    {
        return eval_henyey_greenstein_sampling_pdf(dot(wi, wo), g);
    }
};

/// Evaluates a scalar Henyey-Greenstein phase function.
/// @param cos_theta Dot product of directions that both point away from the scattering point.
/// @param g Mean cosine in the range (-1, 1). Positive values produce forward scattering.
/// @return Phase-function value with respect to solid angle.
float eval_henyey_greenstein(const float cos_theta, const float g)
{
    const float denom = 1 + g * g + 2 * g * cos_theta;
    return float.INV_FOUR_PI * (1 - g * g) / (denom * sqrt(denom));
}

/// Evaluates the scalar PDF used by Henyey-Greenstein sampling.
/// @param cos_theta Dot product of directions that both point away from the scattering point.
/// @param g Mean cosine in the range (-1, 1).
/// @return Sampling density with respect to solid angle.
float eval_henyey_greenstein_sampling_pdf(const float cos_theta, const float g)
{
    return abs(g) < HENYEY_GREENSTEIN_ISOTROPIC_THRESHOLD ? float.INV_FOUR_PI : eval_henyey_greenstein(cos_theta, g);
}

/// Evaluates a spectral Henyey-Greenstein phase function component-wise.
/// @param cos_theta Dot product of directions that both point away from the scattering point.
/// @param g Spectral mean cosine with each component in the range (-1, 1).
/// @return Spectral phase-function value with respect to solid angle.
float3 eval_henyey_greenstein(const float cos_theta, const float3 g)
{
    const float3 denom = 1 + g * g + 2 * g * cos_theta;
    return float.INV_FOUR_PI * (1 - g * g) / (denom * sqrt(denom));
}
