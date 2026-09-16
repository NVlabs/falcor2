// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    const float x = max(1 - cos_theta, 0); // Preserve the endpoint if cos_theta = 1+epsilon.
    const float x2 = x * x;
    const float x5 = x2 * x2 * x;
    return f0 + (f90 - f0) * x5;
}

[Differentiable]
public float eval_fresnel_schlick(float f0, float f90, float cos_theta)
{
    const float x = max(1 - cos_theta, 0); // Preserve the endpoint if cos_theta = 1+epsilon.
    const float x2 = x * x;
    const float x5 = x2 * x2 * x;
    return f0 + (f90 - f0) * x5;
}
