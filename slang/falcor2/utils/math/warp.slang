// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

extension<T> T
    where T : IFloat, IMathConstants
{
    static const T DOMAIN_EPSILON = T.EPSILON * T(1000);
}

// ----------------------------------------------------------------------------
// Spherical coordinates
// ----------------------------------------------------------------------------

/// Converts Cartesian coordinates to spherical coordinates (radians).
/// 'theta' is the polar angle (inclination) between the +z axis and the vector from origin to p,
/// in the range [0,pi].
/// 'phi' is the azimuthal angle from the +x axis in the xy-plane, in the range [0,2pi].
/// @param p Cartesian coordinates (x,y,z).
/// @return Spherical coordinates (theta,phi).
[Differentiable]
public vector<T, 2> cartesian_to_spherical<T>(vector<T, 3> p)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    p = normalize(p);
    vector<T, 2> sph;
    sph.x = acos(p.z);
    sph.y = atan2(-p.y, -p.x) + T.PI;
    return sph;
}

/// Converts spherical coordinates (radians) to Cartesian coordinates.
/// Inverse of cartesian_to_spherical.
/// @param sph Spherical coordinates (theta,phi).
/// @return Cartesian coordinates (x,y,z).
[Differentiable]
public vector<T, 3> spherical_to_cartesian<T>(vector<T, 2> sph)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    vector<T, 3> p;
    p.x = -cos(sph.y - T.PI) * sin(sph.x);
    p.y = -sin(sph.y - T.PI) * sin(sph.x);
    p.z = cos(sph.x);
    return p;
}

/// Converts Cartesian coordinates to spherical coordinates (unsigned normalized).
/// 'theta' is the polar angle (inclination) between the +z axis and the vector from origin to p,
/// normalized to [0,1].
/// 'phi' is the azimuthal angle from the +x axis in the xy-plane, normalized to [0,1].
/// @param p Cartesian coordinates (x,y,z).
/// @return Spherical coordinates (theta,phi).
[Differentiable]
public vector<T, 2> cartesian_to_spherical_unorm<T>(vector<T, 3> p)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    return cartesian_to_spherical(p) * vector<T, 2>(T.INV_PI, T.INV_TWO_PI);
}

// ----------------------------------------------------------------------------
// Latitude / longitude coordinates
// ----------------------------------------------------------------------------

/// Convert world space direction to (u,v) coord in latitude-longitude map (unsigned normalized).
/// The map is centered around the -z axis and wrapping around in clockwise order (left to right).
/// @param dir World space direction (unnormalized).
/// @return Position in latitude-longitude map in [0,1] for each component.
[Differentiable]
public vector<T, 2> world_to_latlong_map<T>(vector<T, 3> dir)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    dir = normalize(dir);
    vector<T, 2> latlong;
    latlong.x = atan2(dir.x, -dir.z) * T.INV_TWO_PI + T(0.5);
    latlong.y = acos(dir.y) * T.INV_PI;
    return latlong;
}

/// Convert a coordinate in latitude-longitude map (unsigned normalized) to a world space direction.
/// The map is centered around the -z axis and wrapping around in clockwise order (left to right).
/// @param latlong Position in latitude-longitude map in [0,1] for each component.
/// @return Normalized direction in world space.
[Differentiable]
public vector<T, 3> latlong_map_to_world<T>(vector<T, 2> latlong)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    T phi = T.PI * (T(2) * saturate(latlong.x) - T(1));
    T theta = T.PI * saturate(latlong.y);
    T sin_theta, cos_theta;
    sincos(theta, sin_theta, cos_theta);
    T sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    return vector<T, 3>(sin_theta * sin_phi, cos_theta, -sin_theta * cos_phi);
}

// ----------------------------------------------------------------------------
// Octahedral mapping
//
// The center of the map represents the +z axis and its corners -z.
// The rotated inner square is the xy-plane projected onto the upper hemi-
// sphere, the outer four triangles folds down over the lower hemisphere.
// There are versions for equal-area and non-equal area (slightly faster).
//
// For details refer to:
// - Clarberg 2008, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD".
// - Cigolle et al. 2014, "Survey of Efficient Representations for Independent Unit Vectors".
//
// ----------------------------------------------------------------------------

/// Helper function to reflect the folds of the lower hemisphere
/// over the diagonals in the octahedral map.
[Differentiable]
public vector<T, 2> oct_wrap<T>(vector<T, 2> v)
    where T : IFloat, __BuiltinFloatingPointType
{
    return (T(1) - abs(v.yx)) * select(v.xy >= T(0), vector<T, 2>(1), vector<T, 2>(-1));
}

/// Converts normalized direction to the octahedral map (non-equal area, signed normalized).
/// @param n Normalized direction.
/// @return Position in octahedral map in [-1,1] for each component.
[Differentiable]
public vector<T, 2> ndir_to_oct_snorm<T>(vector<T, 3> n)
    where T : IFloat, __BuiltinFloatingPointType
{
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1) and then onto the xy-plane.
    vector<T, 2> p = n.xy * (T(1) / (abs(n.x) + abs(n.y) + abs(n.z)));
    p = (n.z < T(0)) ? oct_wrap(p) : p;
    return p;
}

/// Converts normalized direction to the octahedral map (non-equal area, unsigned normalized).
/// @param n Normalized direction.
/// @return Position in octahedral map in [0,1] for each component.
[Differentiable]
public vector<T, 2> ndir_to_oct_unorm<T>(vector<T, 3> n)
    where T : IFloat, __BuiltinFloatingPointType
{
    return ndir_to_oct_snorm(n) * T(0.5) + T(0.5);
}

/// Converts point in the octahedral map to normalized direction (non-equal area, signed normalized).
/// @param p Position in octahedral map in [-1,1] for each component.
/// @return Normalized direction.
[Differentiable]
public vector<T, 3> oct_to_ndir_snorm<T>(vector<T, 2> p)
    where T : IFloat, __BuiltinFloatingPointType
{
    vector<T, 3> n = vector<T, 3>(p.xy, T(1) - abs(p.x) - abs(p.y));
    n.xy = (n.z < T(0)) ? oct_wrap(n.xy) : n.xy;
    return normalize(n);
}

/// Converts point in the octahedral map to normalized direction (non-equal area, unsigned normalized).
/// @param p Position in octahedral map in [0,1] for each component.
/// @return Normalized direction.
[Differentiable]
public vector<T, 3> oct_to_ndir_unorm<T>(vector<T, 2> p)
    where T : IFloat, __BuiltinFloatingPointType
{
    return oct_to_ndir_snorm(p * T(2) - T(1));
}

/// Converts normalized direction to the octahedral map (equal-area, unsigned normalized).
/// @param n Normalized direction.
/// @return Position in octahedral map in [0,1] for each component.
[Differentiable]
public vector<T, 2> ndir_to_oct_equal_area_unorm<T>(vector<T, 3> n)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    // Use atan2 to avoid explicit div-by-zero check in atan(y/x).
    T r = sqrt(T(1) - abs(n.z));
    T phi = atan2(abs(n.y), abs(n.x));

    // Compute p = (u,v) in the first quadrant.
    vector<T, 2> p;
    p.y = r * phi * T.TWO_OVER_PI;
    p.x = r - p.y;

    // Reflect p over the diagonals, and move to the correct quadrant.
    if (n.z < T(0))
        p = T(1) - p.yx;
    p *= vector<T, 2>(sign(n.xy));

    return p * T(0.5) + T(0.5);
}

/// Converts point in the octahedral map to normalized direction (equal area, unsigned normalized).
/// @param p Position in octahedral map in [0,1] for each component.
/// @return Normalized direction.
[Differentiable]
public vector<T, 3> oct_to_ndir_equal_area_unorm<T>(vector<T, 2> p)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    p = p * T(2) - T(1);

    // Compute radius r without branching. The radius r=0 at +z (center) and at -z (corners).
    T d = T(1) - (abs(p.x) + abs(p.y));
    T r = T(1) - abs(d);

    // Compute phi in [0,pi/2] (first quadrant) and sin/cos without branching.
    // TODO: Analyze fp32 precision, do we need a small epsilon instead of 0.0 here?
    T phi = (r > T(0)) ? ((abs(p.y) - abs(p.x)) / r + T(1)) * T.PI_OVER_FOUR : T(0);

    // Convert to Cartesian coordinates. Note that sign(x)=0 for x=0, but that's fine here.
    T f = r * sqrt(T(2) - r * r);
    T x = f * T(sign(p.x)) * cos(phi);
    T y = f * T(sign(p.y)) * sin(phi);
    T z = T(sign(d)) * (T(1) - r * r);

    return vector<T, 3>(x, y, z);
}

// ----------------------------------------------------------------------------
// Uniform disk (polar, high-distortion)
// ----------------------------------------------------------------------------

/// Uniformly sample a 2D disk w.r.t. unit area.
/// @param sample 2D sample in [0,1)^2.
/// @return 2D point on the unit disk.
[Differentiable]
public vector<T, 2> square_to_uniform_disk_polar<T>(vector<T, 2> sample)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    T r = sqrt(sample.y);
    T s, c;
    sincos(sample.x * T.TWO_PI, s, c);
    return vector<T, 2>(c, s) * r;
}

/// Inverse of the mapping @ref square_to_uniform_disk_polar.
/// @param p 2D point on the unit disk.
/// @return 2D sample in [0,1)^2.
[Differentiable]
public vector<T, 2> uniform_disk_to_square_polar<T>(vector<T, 2> p)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    T phi = atan2(p.y, p.x) * T.INV_TWO_PI;
    return vector<T, 2>(select(phi < T(0), phi + T(1), phi), squared_norm(p));
}

/// Density of @ref square_to_uniform_disk_polar w.r.t. unit area.
/// @param p A 2D point on the unit disk.
/// @param test_domain If true, returns 0 outside the unit disk.
/// @return The density at the given point.
[Differentiable]
public T square_to_uniform_disk_polar_pdf<T>(vector<T, 2> p, bool test_domain = false)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    if (test_domain && squared_norm(p) > T(1))
        return T(0);
    return T.INV_PI;
}

// ----------------------------------------------------------------------------
// Uniform disk (concentric, low-distortion)
// ----------------------------------------------------------------------------

/// Uniformly sample a 2D disk w.r.t. unit area.
/// Low-distortion concentric square to disk mapping by Peter Shirley.
/// @param sample 2D sample in [0,1)^2.
/// @return 2D point on the unit disk.
[Differentiable]
public vector<T, 2> square_to_uniform_disk_concentric<T>(vector<T, 2> sample)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    vector<T, 2> t = T(2) * sample - T(1);

    bool is_zero = all(t == T(0));
    bool quadrant_1_or_3 = abs(t.x) < abs(t.y);

    T r = select(quadrant_1_or_3, t.y, t.x);
    T rp = select(quadrant_1_or_3, t.x, t.y);

    T phi = T(0.25) * T.PI * rp / r;
    phi = select(quadrant_1_or_3, T(0.5) * T.PI - phi, phi);
    phi = select(is_zero, T(0), phi);

    T s, c;
    sincos(phi, s, c);
    return vector<T, 2>(c, s) * r;
}

/// Inverse of the mapping @ref square_to_uniform_disk_concentric.
/// @param p 2D point on the unit disk.
/// @return 2D sample in [0,1)^2.
[Differentiable]
public vector<T, 2> uniform_disk_to_square_concentric<T>(vector<T, 2> p)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    bool quadrant_0_or_2 = abs(p.x) > abs(p.y);
    T r_sign = select(quadrant_0_or_2, p.x, p.y);
    T r = copysign(length(p), r_sign);

    r_sign = select(r_sign >= T(0), T(1), T(-1));
    T phi = atan2(p.y * r_sign, p.x * r_sign);

    T t = T(4) / T.PI * phi;
    t = select(quadrant_0_or_2, t, T(2) - t) * r;

    T a = select(quadrant_0_or_2, r, t);
    T b = select(quadrant_0_or_2, t, r);

    return (vector<T, 2>(a, b) + T(1)) * T(0.5);
}

/// Density of @ref square_to_uniform_disk_concentric w.r.t. unit area.
/// @param p 2D point on the unit disk.
/// @param test_domain If true, returns 0 outside the unit disk.
/// @return The density at the given point.
[Differentiable]
T square_to_uniform_disk_concentric_pdf<T>(vector<T, 2> p, bool test_domain = false)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    if (test_domain && squared_norm(p) > T(1))
        return T(0);
    return T.INV_PI;
}

// ----------------------------------------------------------------------------
// Uniform sphere
// ----------------------------------------------------------------------------

/// Uniformly sample a vector on the unit sphere w.r.t. solid angle.
/// @param sample 2D sample in [0,1)^2.
/// @return 3D vector on the unit sphere.
[Differentiable]
public vector<T, 3> square_to_uniform_sphere<T>(vector<T, 2> sample)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    T phi = sample.x * T.TWO_PI;
    T cos_theta = T(1) - T(2) * sample.y;
    T sin_theta = sqrt(max(T(0), T(1) - cos_theta * cos_theta));
    T sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    return vector<T, 3>(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

/// Inverse of the mapping @ref square_to_uniform_sphere.
/// @param p 3D vector on the unit sphere.
/// @return 2D sample in [0,1)^2.
[Differentiable]
public vector<T, 2> uniform_sphere_to_square<T>(vector<T, 3> p)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    T phi = atan2(p.y, p.x) * T.INV_TWO_PI;
    return vector<T, 2>(select(phi < T(0), phi + T(1), phi), T(0.5) * (T(1) - p.z));
}

/// Density of @ref square_to_uniform_sphere w.r.t. solid angle.
/// @param p 3D vector on the unit sphere.
/// @param test_domain If true, returns 0 outside the unit sphere.
/// @return The density at the given vector.
[Differentiable]
public T square_to_uniform_sphere_pdf<T>(vector<T, 3> p, bool test_domain = false)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    if (test_domain && abs(squared_norm(p) - T(1)) > T.DOMAIN_EPSILON)
        return T(0);
    return T.INV_FOUR_PI;
}

// ----------------------------------------------------------------------------
// Uniform hemisphere (polar, high-distortion)
// ----------------------------------------------------------------------------

/// Uniformly sample a vector on the unit hemisphere w.r.t. solid angle.
/// @param sample 2D sample in [0,1)^2.
/// @return 3D vector on the unit hemisphere.
[Differentiable]
public vector<T, 3> square_to_uniform_hemisphere_polar<T>(vector<T, 2> sample)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    T phi = sample.x * T.TWO_PI;
    T cos_theta = sample.y;
    T sin_theta = safe_sqrt(T(1) - sqr(cos_theta));
    T sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    return vector<T, 3>(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

/// Inverse of the mapping @ref square_to_uniform_hemisphere_polar.
/// @param p 3D vector on the unit hemisphere.
/// @return 2D sample in [0,1)^2.
[Differentiable]
public vector<T, 2> uniform_hemisphere_to_square_polar<T>(vector<T, 3> p)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    T phi = atan2(p.y, p.x) * T.INV_TWO_PI;
    return vector<T, 2>(select(phi < T(0), phi + T(1), phi), p.z);
}

/// Density of @ref square_to_uniform_hemisphere_polar w.r.t. solid angle.
/// @param p 3D vector on the unit hemisphere.
/// @param test_domain If true, returns 0 outside the unit hemisphere.
/// @return The density at the given vector.
[Differentiable]
public T square_to_uniform_hemisphere_polar_pdf<T>(vector<T, 3> p, bool test_domain = false)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    if (test_domain && (abs(squared_norm(p) - T(1)) > T.DOMAIN_EPSILON || p.z < T(0)))
        return T(0);
    return T.INV_TWO_PI;
}

// ----------------------------------------------------------------------------
// Uniform hemisphere (concentric, low-distortion)
// ----------------------------------------------------------------------------

/// Uniformly sample a vector on the unit hemisphere w.r.t. solid angle.
/// @param sample 2D sample in [0,1)^2.
/// @return 3D vector on the unit hemisphere.
[Differentiable]
public vector<T, 3> square_to_uniform_hemisphere_concentric<T>(vector<T, 2> sample)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    vector<T, 2> p = square_to_uniform_disk_concentric(sample);
    T z = T(1) - squared_norm(p);
    p *= sqrt(z + T(1));
    return vector<T, 3>(p, z);
}

/// Inverse of the mapping @ref square_to_uniform_hemisphere_concentric.
/// @param p 3D vector on the unit hemisphere.
/// @return 2D sample in [0,1)^2.
[Differentiable]
public vector<T, 2> uniform_hemisphere_to_square_concentric<T>(vector<T, 3> p)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    return uniform_disk_to_square_concentric(p.xy * rsqrt(p.z + T(1)));
}

/// Density of @ref square_to_uniform_hemisphere_concentric w.r.t. solid angle.
/// @param p 3D vector on the unit hemisphere.
/// @param test_domain If true, returns 0 outside the unit hemisphere.
/// @return The density at the given vector.
[Differentiable]
public T square_to_uniform_hemisphere_concentric_pdf<T>(vector<T, 3> p, bool test_domain = false)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    if (test_domain && (abs(squared_norm(p) - T(1)) > T.DOMAIN_EPSILON || p.z < T(0)))
        return T(0);
    return T.INV_TWO_PI;
}

// ----------------------------------------------------------------------------
// Cosine-weighted hemisphere (polar, high-distortion)
// ----------------------------------------------------------------------------

/// Sample a cosine-weighted vector on the unit hemisphere w.r.t. solid angle.
/// @param sample 2D sample in [0,1)^2.
/// @return 3D vector on the unit hemisphere.
[Differentiable]
public vector<T, 3> square_to_cosine_hemisphere_polar<T>(vector<T, 2> sample)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    vector<T, 3> p;
    T r = sqrt(sample.y);
    T phi = sample.x * T.TWO_PI;
    sincos(phi, p.y, p.x);
    p.xy *= r;
    p.z = sqrt(T(1) - sample.y);
    return p;
}

/// Inverse of the mapping @ref square_to_cosine_hemisphere_polar.
/// @param p 3D vector on the unit hemisphere.
/// @return 2D sample in [0,1)^2.
[Differentiable]
public vector<T, 2> cosine_hemisphere_to_square_polar<T>(vector<T, 3> p)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    T phi = atan2(p.y, p.x) * T.INV_TWO_PI;
    return vector<T, 2>(select(phi < T(0), phi + T(1), phi), T(1) - sqr(p.z));
}

/// Density of @ref square_to_cosine_hemisphere_polar() w.r.t solid angle.
/// @param p 3D vector on the unit hemisphere.
/// @param test_domain If true, returns 0 outside the unit hemisphere.
/// @return The density at the given vector.
[Differentiable]
public T square_to_cosine_hemisphere_polar_pdf<T>(vector<T, 3> p, bool test_domain = false)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    if (test_domain && (abs(squared_norm(p) - T(1)) > T.DOMAIN_EPSILON || p.z < T(0)))
        return T(0);
    return T.INV_PI * p.z;
}

// ----------------------------------------------------------------------------
// Cosine-weighted hemisphere (low-distortion)
// ----------------------------------------------------------------------------

/// Sample a cosine-weighted vector on the unit hemisphere w.r.t. solid angle.
/// @param sample 2D sample in [0,1)^2.
/// @return 3D vector on the unit hemisphere.
[Differentiable]
public vector<T, 3> square_to_cosine_hemisphere_concentric<T>(vector<T, 2> sample)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    vector<T, 2> p = square_to_uniform_disk_concentric(sample);
    return vector<T, 3>(p, safe_sqrt(T(1) - squared_norm(p)));
}

/// Inverse of the mapping @ref square_to_cosine_hemisphere_concentric.
/// @param p 3D vector on the unit hemisphere.
/// @return 2D sample in [0,1)^2.
[Differentiable]
public vector<T, 2> cosine_hemisphere_to_square_concentric<T>(vector<T, 3> p)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    return uniform_disk_to_square_concentric(p.xy);
}

/// Density of @ref square_to_cosine_hemisphere_concentric() w.r.t solid angle.
/// @param p 3D vector on the unit hemisphere.
/// @param test_domain If true, returns 0 outside the unit hemisphere.
/// @return The density at the given vector.
[Differentiable]
public T square_to_cosine_hemisphere_concentric_pdf<T>(vector<T, 3> p, bool test_domain = false)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    if (test_domain && (abs(squared_norm(p) - T(1)) > T.DOMAIN_EPSILON || p.z < T(0)))
        return T(0);
    return T.INV_PI * p.z;
}

// ----------------------------------------------------------------------------
// Cone of angles (concentric, low-distortion)
// ----------------------------------------------------------------------------

/// Uniformly sample a vector that lies within a cone of angles around the Z axis.
/// @param sample 2D sample in [0,1)^2.
/// @param cos_cutoff Cosine of cutoff angle.
/// @return 3D vector within the cone of angles.
[Differentiable]
public vector<T, 3> square_to_uniform_cone_concentric<T>(vector<T, 2> sample, T cos_cutoff)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    T one_minus_cos_cutoff = T(1) - cos_cutoff;
    vector<T, 2> p = square_to_uniform_disk_concentric(sample);
    T pn = squared_norm(p);
    T z = cos_cutoff + one_minus_cos_cutoff * (T(1) - pn);
    p *= safe_sqrt(one_minus_cos_cutoff * (T(2) - one_minus_cos_cutoff * pn));
    return vector<T, 3>(p.x, p.y, z);
}

/// Inverse of the mapping @ref square_to_uniform_cone_concentric.
/// @param v 3D vector within the cone of angles.
/// @param cos_cutoff Cosine of cutoff angle.
/// @return 2D sample in [0,1)^2.
[Differentiable]
public vector<T, 2> uniform_cone_to_square_concentric<T>(vector<T, 3> v, T cos_cutoff)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    vector<T, 2> p = v.xy;
    p *= sqrt((T(1) - v.z) / (squared_norm(p) * (T(1) - cos_cutoff)));
    return uniform_disk_to_square_concentric(p);
}

/// Density of @ref square_to_uniform_cone_concentric() w.r.t solid angle.
/// @param v 3D vector within the cone of angles.
/// @param test_domain If true, returns 0 outside the cone of angles.
/// @return The density at the given vector.
[Differentiable]
public T square_to_uniform_cone_concentric_pdf<T>(vector<T, 3> v, T cos_cutoff, bool test_domain = false)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    if (test_domain && (abs(squared_norm(v) - T(1)) > T.DOMAIN_EPSILON || v.z < cos_cutoff))
        return T(0);
    return T.INV_TWO_PI / (T(1) - cos_cutoff);
}

// ----------------------------------------------------------------------------
// Triangle barycentrics
// ----------------------------------------------------------------------------

/// Uniform sampling of a triangle.
/// @param u Uniform random numbers in [0,1)^2.
/// @return Barycentric coordinates (1-u-v,u,v) of the sampled point.
[Differentiable]
public vector<T, 3> sample_triangle<T>(vector<T, 2> u)
    where T : IFloat, __BuiltinFloatingPointType, IMathConstants
{
    T su = sqrt(u.x);
    vector<T, 2> b = vector<T, 2>(T(1) - su, u.y * su);
    return vector<T, 3>(T(1) - b.x - b.y, b.x, b.y);
}
