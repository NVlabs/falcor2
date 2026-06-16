// SPDX-License-Identifier: Apache-2.0

implementing utils;


/// Evaluates the Fresnel term using Schlick's approximation.
/// Introduced in http://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf
///
/// The Fresnel term equals f0 at normal incidence, and approaches f90=1.0 at 90 degrees.
/// The formulation below is generalized to allow both f0 and f90 to be specified.
///
/// @param f0 Specular reflectance at normal incidence (0 degrees).
/// @param f90 Reflectance at orthogonal incidence (90 degrees), which should be 1.0 for specular surface reflection.
/// @param cos_theta Cosine of angle between microfacet normal and incident direction.
/// @return Fresnel term.
[Differentiable]
public float3 eval_fresnel_schlick(float3 f0, float3 f90, float cos_theta)
{
    return f0 + (f90 - f0) * pow(max(1 - cos_theta, 0), 5); // Clamp to avoid NaN if cos_theta = 1+epsilon
}

[Differentiable]
public float eval_fresnel_schlick(float f0, float f90, float cos_theta)
{
    return f0 + (f90 - f0) * pow(max(1 - cos_theta, 0), 5); // Clamp to avoid NaN if cos_theta = 1+epsilon
}
