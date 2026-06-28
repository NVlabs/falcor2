// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Evaluates the Fresnel term using Schlick's approximation.
/// Introduced in http://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf
///
/// The Fresnel term equals f0 at normal incidence, and approaches f90=1.0 at 90 degrees.
/// The formulation below is generalized to allow both f0 and f90 to be specified.
///
/// @param f0 Specular reflectance at normal incidence (0 degrees).
/// @param f90 Reflectance at orthogonal incidence (90 degrees), which should be 1.0 for specular surface reflection.
/// @param cos_theta Cosine of angle between microfacet normal and incident direction (LdotH).
/// @return Fresnel term.
[Differentiable]
float3 eval_fresnel_schlick(float3 f0, float3 f90, float cos_theta)
{
    return f0 + (f90 - f0) * pow(max(1.0 - cos_theta, 0.0), 5.0); // Clamp to avoid NaN if cos_theta = 1+epsilon
}

[Differentiable]
float eval_fresnel_schlick(float f0, float f90, float cos_theta)
{
    return f0 + (f90 - f0) * pow(max(1.0 - cos_theta, 0.0), 5.0); // Clamp to avoid NaN if cos_theta = 1+epsilon
}

[Differentiable]
float3 eval_fresnel_generalized_schlick(float3 f0, float3 f90, float exponent, float cos_theta)
{
    return f0 + (f90 - f0) * pow(max(1.0 - cos_theta, 0.0), exponent); // Clamp to avoid NaN if cos_theta = 1+epsilon
}

/// Evaluates the Fresnel term using dieletric fresnel equations.
/// Based on http://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission.html
///
/// @param eta Relative index of refraction (etaI / etaT).
/// @param cos_theta_i Cosine of angle between normal and incident direction.
/// @param[out] cos_theta_t Cosine of angle between negative normal and transmitted direction (0 for total internal
/// reflection).
/// @return Returns Fr(eta, cos_theta_i).
[Differentiable]
float eval_fresnel_dielectric(float eta, float cos_theta_i, out float cos_theta_t)
{
    if (cos_theta_i < 0.0) {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
    }

    float sin_theta_t_sq = eta * eta * (1.0 - cos_theta_i * cos_theta_i);
    // Check for total internal reflection
    if (sin_theta_t_sq > 1.0) {
        cos_theta_t = 0.0;
        return 1.0;
    }

    cos_theta_t = sqrt(1.0 - sin_theta_t_sq); // No clamp needed

    // Note that at eta=1 and cos_theta_i=0, we get cos_theta_t=0 and NaN below.
    // It's important the framework clamps |cos_theta_i| or eta to small epsilon.
    float Rs = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    float Rp = (eta * cos_theta_t - cos_theta_i) / (eta * cos_theta_t + cos_theta_i);

    return 0.5 * (Rs * Rs + Rp * Rp);
}

/// Evaluates the Fresnel term using dieletric fresnel equations.
/// Based on http://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission.html
///
/// @param eta Relative index of refraction (etaI / etaT).
/// @param cos_theta_i Cosine of angle between normal and incident direction.
/// @return Returns Fr(eta, cos_theta_i).
[Differentiable]
float eval_fresnel_dielectric(float eta, float cos_theta_i)
{
    float cos_theta_t;
    return eval_fresnel_dielectric(eta, cos_theta_i, cos_theta_t);
}

/// Evaluates the Fresnel term using conductor fresnel equations, assuming unpolarized light.
/// Base on "PHYSICALLY BASED LIGHTING CALCULATIONS FOR COMPUTER GRAPHICS" by Peter Shirley
/// http://www.cs.virginia.edu/~jdl/bib/globillum/shirley_thesis.pdf
///
/// @param eta Real part of complex index of refraction
/// @param k Imaginary part of complex index of refraction (the "absorption coefficient")
/// @param cos_theta_i Cosine of angle between normal and incident direction.
/// @return Returns conductor reflectance.
[Differentiable]
float eval_fresnel_conductor(float eta, float k, float cos_theta_i)
{
    float cos_theta_i_sq = cos_theta_i * cos_theta_i;
    float sin_theta_i_sq = max(1.0 - cos_theta_i_sq, 0.0);
    float sin_theta_i_qu = sin_theta_i_sq * sin_theta_i_sq;

    float inner_term = eta * eta - k * k - sin_theta_i_sq;
    float a_sq_plus_b_sq = sqrt(max(inner_term * inner_term + 4.0 * eta * eta * k * k, 0.0));
    float a = sqrt(max((a_sq_plus_b_sq + inner_term) * 0.5, 0.0));

    float Rs = ((a_sq_plus_b_sq + cos_theta_i_sq) - (2.0 * a * cos_theta_i))
        / ((a_sq_plus_b_sq + cos_theta_i_sq) + (2.0 * a * cos_theta_i));
    float Rp = ((cos_theta_i_sq * a_sq_plus_b_sq + sin_theta_i_qu) - (2.0 * a * cos_theta_i * sin_theta_i_sq))
        / ((cos_theta_i_sq * a_sq_plus_b_sq + sin_theta_i_qu) + (2.0 * a * cos_theta_i * sin_theta_i_sq));

    return 0.5 * (Rs + Rs * Rp);
}

/// Evaluates the Fresnel term using conductor fresnel equations, assuming unpolarized light.
/// Convenience function that takes coefficients at 3 wavelengths.
///
/// @param eta Real part of complex index of refraction
/// @param k Imaginary part of complex index of refraction (the "absorption coefficient")
/// @param cos_theta_i Cosine of angle between normal and incident direction.
/// @return Returns conductor reflectance.
[Differentiable]
float3 eval_fresnel_conductor(float3 eta, float3 k, float cos_theta_i)
{
    return float3(
        eval_fresnel_conductor(eta.x, k.x, cos_theta_i),
        eval_fresnel_conductor(eta.y, k.y, cos_theta_i),
        eval_fresnel_conductor(eta.z, k.z, cos_theta_i)
    );
}

/// Converts reflectance into IOR assuming normal incidence and n_i = 1.0
///
/// @param reflectance Reflectance at normal incidence
/// @return Returns index or refraction.
public float reflectance_to_ior(float3 reflectance)
{
    const float r = min(luminance(reflectance), 0.999);
    const float ior_i = 1.0;

    return (-ior_i - 2.0 * ior_i * sqrt(r) - ior_i * r) / (r - 1.0);
}
