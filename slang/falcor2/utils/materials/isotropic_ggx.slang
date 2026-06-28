// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;


/// Evaluates the GGX (Trowbridge-Reitz) normal distribution function (D).
///
/// Introduced by Trowbridge and Reitz, "Average irregularity representation of a rough surface for ray reflection",
/// Journal of the Optical Society of America, vol. 65(5), 1975. See the correct normalization factor in Walter et al.
/// https://dl.acm.org/citation.cfm?id=2383874 We use the simpler, but equivalent expression in Eqn 19 from
/// http://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf
///
/// For microfacet models, D is evaluated for the direction h to find the density of potentially active microfacets
/// (those for which microfacet normal m = h). The 'alpha' parameter is the standard GGX width, e.g., it is the square
/// of the linear roughness parameter in Disney's BRDF. Note there is a singularity (0/0 = NaN) at NdotH = 1 and
/// alpha = 0, so alpha should be clamped to some epsilon.
///
/// @param alpha GGX width parameter (should be clamped to small epsilon beforehand).
/// @param cosTheta Dot product between shading normal and half vector, in positive hemisphere.
/// @return D(h)
[Differentiable]
public float evalNdfGGX(float alpha, float cosTheta)
{
    float a2 = alpha * alpha;
    float d = ((cosTheta * a2 - cosTheta) * cosTheta + 1);
    return a2 / (d * d * float.PI);
}

/// Evaluates the PDF for sampling the GGX normal distribution function using Walter et al. 2007's method.
/// See https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
///
/// @param alpha GGX width parameter (should be clamped to small epsilon beforehand).
/// @param cosTheta Dot product between shading normal and half vector, in positive hemisphere.
/// @return D(h) * cosTheta
[Differentiable]
public float evalPdfGGX_NDF(float alpha, float cosTheta)
{
    return evalNdfGGX(alpha, cosTheta) * cosTheta;
}

/// Samples the GGX (Trowbridge-Reitz) normal distribution function (D) using Walter et al. 2007's method.
/// Note that the sampled half vector may lie in the negative hemisphere. Such samples should be discarded.
/// See Eqn 35 & 36 in https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
/// See Listing A.1 in https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
///
/// @param alpha GGX width parameter (should be clamped to small epsilon beforehand).
/// @param u Uniform random number (2D).
/// @param[out] pdf Sampling probability.
/// @return Sampled half vector in local space.
[Differentiable]
public float3 sampleGGX_NDF(float alpha, float2 u, out float pdf)
{
    float alphaSqr = alpha * alpha;
    float phi = u.y * (2 * float.PI);
    float tanThetaSqr = alphaSqr * u.x / (1 - u.x);
    float cosTheta = 1 / sqrt(1 + tanThetaSqr);
    float r = sqrt(max(1 - cosTheta * cosTheta, 0));

    pdf = evalPdfGGX_NDF(alpha, cosTheta);
    return float3(cos(phi) * r, sin(phi) * r, cosTheta);
}

/// Evaluates the PDF for sampling the GGX distribution of visible normals (VNDF).
/// See http://jcgt.org/published/0007/04/01/paper.pdf
///
/// @param alpha GGX width parameter (should be clamped to small epsilon beforehand).
/// @param wi Incident direction in local space, in the positive hemisphere.
/// @param h Half vector in local space, in the positive hemisphere.
/// @return D_V(h) = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
[Differentiable]
public float evalPdfGGX_VNDF(float alpha, float3 wi, float3 h)
{
    float G1 = evalG1GGX(alpha * alpha, wi.z);
    float D = evalNdfGGX(alpha, h.z);
    return G1 * D * max(0.f, dot(wi, h)) / wi.z;
}

[Differentiable]
public float3 SampleVndf_Hemisphere(float2 u, float3 Vh)
{
    float phi = 2.f * float.PI * u.x;
    float z = fma((1.f - u.y), (1.f + Vh.z), -Vh.z);
    float sinTheta = sqrt(saturate(1.f - z * z));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float3 c = float3(x, y, z);
    float3 h = c + Vh;
    return h;
}

/// Samples the GGX (Trowbridge-Reitz) using the distribution of visible normals (VNDF).
/// The GGX VDNF yields significant variance reduction compared to sampling of the GGX NDF.
/// See http://jcgt.org/published/0007/04/01/paper.pdf
/// The visible hemisphere sampling procedure is now performed with the technique in
/// "Sampling Visible GGX Normals with Spherical Caps" by Dupuy and Benyoub [2023]
/// See https://arxiv.org/abs/2306.05044
/// @param alpha Isotropic GGX width parameter (should be clamped to small epsilon beforehand).
/// @param wi Incident direction in local space, in the positive hemisphere.
/// @param u Uniform random number (2D).
/// @param[out] pdf Sampling probability.
/// @return Sampled half vector in local space, in the positive hemisphere.
[Differentiable]
public float3 sampleGGX_VNDF(float alpha, float3 wi, float2 u, out float pdf)
{
    float alpha_x = alpha, alpha_y = alpha;
    // Transform the view vector to the hemisphere configuration.
    float3 Vh = normalize(float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));
    float3 Nh = SampleVndf_Hemisphere(u, Vh);
    // Transform the normal back to the ellipsoid configuration. This is our half vector.
    float3 h = normalize(float3(alpha_x * Nh.x, alpha_y * Nh.y, Nh.z));
    pdf = evalPdfGGX_VNDF(alpha, wi, h);
    return h;
}

/// Evaluates the Smith masking function (G1) for the GGX normal distribution.
/// See Eq 34 in https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
///
/// The evaluated direction is assumed to be in the positive hemisphere relative the half vector.
/// This is the case when both incident and outgoing direction are in the same hemisphere, but care should be taken
/// with transmission.
///
/// @param alphaSqr Squared GGX width parameter.
/// @param cosTheta Dot product between shading normal and evaluated direction, in the positive hemisphere.
[Differentiable]
public float evalG1GGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0)
        return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0) / cosThetaSqr;
    return 2 / (1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

/// Evaluates the Smith lambda function for the GGX normal distribution.
/// See Eq 72 in http://jcgt.org/published/0003/02/03/paper.pdf
///
/// @param alphaSqr Squared GGX width parameter.
/// @param cosTheta Dot product between shading normal and the evaluated direction, in the positive hemisphere.
[Differentiable]
public float evalLambdaGGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0)
        return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0) / cosThetaSqr;
    return 0.5 * (-1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

/// Evaluates the separable form of the masking-shadowing function for the GGX normal distribution, using Smith's
/// approximation. See Eq 98 in http://jcgt.org/published/0003/02/03/paper.pdf
///
/// @param alpha GGX width parameter (should be clamped to small epsilon beforehand).
/// @param cosThetaI Dot product between shading normal and incident direction, in positive hemisphere.
/// @param cosThetaO Dot product between shading normal and outgoing direction, in positive hemisphere.
/// @return G(cosThetaI, cosThetaO)
[Differentiable]
public float evalMaskingSmithGGXSeparable(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1 / ((1 + lambdaI) * (1 + lambdaO));
}

/// Evaluates the height-correlated form of the masking-shadowing function for the GGX normal distribution, using
/// Smith's approximation. See Eq 99 in http://jcgt.org/published/0003/02/03/paper.pdf
///
/// Eric Heitz recommends using it in favor of the separable form as it is more accurate and of similar complexity.
/// The function is only valid for cosThetaI > 0 and cosThetaO > 0  and should be clamped to 0 otherwise.
///
/// @param alpha GGX width parameter (should be clamped to small epsilon beforehand).
/// @param cosThetaI Dot product between shading normal and incident direction, in positive hemisphere.
/// @param cosThetaO Dot product between shading normal and outgoing direction, in positive hemisphere.
/// @return G(cosThetaI, cosThetaO)
[Differentiable]
public float evalMaskingSmithGGXCorrelated(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1 / (1 + lambdaI + lambdaO);
}
