// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Ray-sphere intersection.
/// This function implements the standard analytic test with improvements to floating-point precision
/// and returns the closest hit.
/// @param ray_origin Ray origin.
/// @param ray_dir Ray direction (does not have to be normalized).
/// @param sphere_center Sphere center.
/// @param sphere_radius Sphere radius.
/// @param[out] t Distance to the closest intersection.
/// @param t_min Minimum t value (defaults to 0.0).
/// @param t_max Maximum t value (defaults to float.MAX).
/// @param backface_cull Whether to cull backfaces (defaults to false).
/// @return True if the ray intersects the sphere.
public bool intersect_ray_sphere(
    const float3 ray_origin,
    const float3 ray_dir,
    const float3 sphere_center,
    const float sphere_radius,
    out float t,
    const float t_min = 0.0,
    const float t_max = float.MAX,
    const bool backface_cull = false
)
{
    t = {};

    // Implementation is taken from Chapter 7 of Ray-Tracing Gems
    float3 f = ray_origin - sphere_center;
    float a = dot(ray_dir, ray_dir);
    float b = dot(-f, ray_dir);
    float discriminant = sphere_radius * sphere_radius - dot(f + b / a * ray_dir, f + b / a * ray_dir);

    // Negative discriminant means ray missed sphere.
    if (discriminant < 0.0)
        return false;

    // If b and discriminant are both 0, then the ray's origin lies on the sphere
    if (b == 0.0 && discriminant == 0.0 && t_min <= 0.0) {
        t = 0.0;
        return (t <= t_max);
    }

    // There are two solutions t0 and t1, but one or both may be negative.
    float c = dot(f, f) - sphere_radius * sphere_radius;
    float signB = (b < 0.0) ? -1.0 : 1.0;
    float q = b + signB * sqrt(a * discriminant);
    float t0 = c / q;
    float t1 = q / a;

    float tc = (t0 < t_min && !backface_cull) ? t1 : t0; // tc is the closest hit we care about
    if (tc < t_min)
        return false;

    t = tc;
    return (t <= t_max);
}

/// Ray intersection against the lateral surface of a finite open circular cylinder.
/// @param ray_origin Ray origin.
/// @param ray_dir Ray direction (does not have to be normalized).
/// @param cylinder_center Cylinder center.
/// @param cylinder_axis Unit vector along the cylinder axis.
/// @param cylinder_radius Cylinder radius.
/// @param cylinder_half_length Cylinder half-length along cylinder_axis.
/// @param[out] t Distance to the closest intersection.
/// @param t_min Minimum t value (defaults to 0.0).
/// @param t_max Maximum t value (defaults to float.MAX).
/// @return True if the ray intersects the cylinder's lateral surface.
public bool intersect_ray_open_cylinder(
    const float3 ray_origin,
    const float3 ray_dir,
    const float3 cylinder_center,
    const float3 cylinder_axis,
    const float cylinder_radius,
    const float cylinder_half_length,
    out float t,
    const float t_min = 0.0,
    const float t_max = float.MAX
)
{
    t = {};

    const float3 offset = ray_origin - cylinder_center;
    const float origin_axial = dot(offset, cylinder_axis);
    const float direction_axial = dot(ray_dir, cylinder_axis);
    const float3 origin_radial = offset - origin_axial * cylinder_axis;
    const float3 direction_radial = ray_dir - direction_axial * cylinder_axis;
    const float a = dot(direction_radial, direction_radial);
    if (a <= float.MIN)
        return false;

    const float t_center = -dot(origin_radial, direction_radial) / a;
    const float3 closest_radial = origin_radial + t_center * direction_radial;
    const float discriminant = cylinder_radius * cylinder_radius - dot(closest_radial, closest_radial);
    if (discriminant < 0.0)
        return false;

    const float t_offset = safe_sqrt(discriminant / a);
    const float t0 = t_center - t_offset;
    const float t1 = t_center + t_offset;
    if (t0 >= t_min && t0 <= t_max && abs(origin_axial + t0 * direction_axial) <= cylinder_half_length) {
        t = t0;
        return true;
    }
    if (t1 >= t_min && t1 <= t_max && abs(origin_axial + t1 * direction_axial) <= cylinder_half_length) {
        t = t1;
        return true;
    }
    return false;
}

/// Ray-AABB intersection.
/// @param ray_origin Ray origin.
/// @param ray_dir Ray direction (does not have to be normalized).
/// @param aabb_min AABB minimum.
/// @param aabb_max AABB maximum.
/// @param[out] near_far Returns intersection interval along ray.
/// @return True if the ray intersects the AABB.
public bool intersect_ray_aabb(
    const float3 ray_origin,
    const float3 ray_dir,
    const float3 aabb_min,
    const float3 aabb_max,
    out float2 near_far
)
{
    const float3 inv_dir = 1.0 / ray_dir;
    const float3 lo = (aabb_min - ray_origin) * inv_dir;
    const float3 hi = (aabb_max - ray_origin) * inv_dir;
    const float3 tmin = min(lo, hi), tmax = max(lo, hi);
    near_far.x = max(0.0, max(tmin.x, max(tmin.y, tmin.z)));
    near_far.y = min(tmax.x, min(tmax.y, tmax.z));
    return near_far.x <= near_far.y;
}

/// Ray-Triangle intersection based on "Watertight Ray/Triangle Intersection"
/// Paper link: http://jcgt.org/published/0002/01/05/paper.pdf
/// @param ray_origin Ray origin.
/// @param ray_dir Ray direction (does not have to be normalized).
/// @param vertices Triangle vertices.
/// @param[out] barycentrics Barycentric coordinates.
/// @param[out] t Intersection distance.
/// @return True if the ray intersects the triangle.
public bool intersect_ray_triangle(
    const float3 ray_origin,
    const float3 ray_dir,
    const float3 vertices[3],
    out float3 barycentrics,
    out float t
)
{
    barycentrics = {};
    t = {};

    const float3 abs_dir = abs(ray_dir);
    uint axis = 0;
    if (abs_dir.y > abs_dir.x && abs_dir.y > abs_dir.z)
        axis = 1;
    if (abs_dir.z > abs_dir.x && abs_dir.z > abs_dir.y)
        axis = 2;

    uint kz = axis;
    uint kx = (kz + 1) % 3;
    uint ky = (kx + 1) % 3;
    if (ray_dir[kz] < 0.0) {
        uint swap = kx;
        kx = ky;
        ky = swap;
    }

    float Sx = ray_dir[kx] / ray_dir[kz];
    float Sy = ray_dir[ky] / ray_dir[kz];
    float Sz = 1.0 / ray_dir[kz];

    const float3 A = vertices[0] - ray_origin;
    const float3 B = vertices[1] - ray_origin;
    const float3 C = vertices[2] - ray_origin;

    const float Ax = A[kx] - Sx * A[kz];
    const float Ay = A[ky] - Sy * A[kz];
    const float Bx = B[kx] - Sx * B[kz];
    const float By = B[ky] - Sy * B[kz];
    const float Cx = C[kx] - Sx * C[kz];
    const float Cy = C[ky] - Sy * C[kz];

    float U = Cx * By - Cy * Bx;
    float V = Ax * Cy - Ay * Cx;
    float W = Bx * Ay - By * Ax;

    if (U == 0.0 || V == 0.0 || W == 0.0) {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);
    }

    if ((U < 0.0 || V < 0.0 || W < 0.0) && (U > 0.0 || V > 0.0 || W > 0.0))
        return false;

    float det = U + V + W;
    if (det == 0.0)
        return false;

    const float Az = Sz * A[kz];
    const float Bz = Sz * B[kz];
    const float Cz = Sz * C[kz];
    const float T = U * Az + V * Bz + W * Cz;
    const float inv_det = 1.0 / det;

    t = T * inv_det;
    barycentrics = float3(U, V, W) * inv_det;

    return true;
}

/// Ray intersection against a linear swept sphere using Han et al. (2019), "Ray Tracing Generalized Tube
/// Primitives: Method and Applications", Sections 3.2.2 and 4.1, https://doi.org/10.1111/cgf.13703.
/// Uses the tangent-cone construction and unit-cone transform, with a cylinder for equal radii.
/// Returns the nearest component hit, not the whole-tube CSG boundary of Section 4.3.
/// Scene traversal uses intersect_procedural_lss in falcor2.render for native LSS entry/cap policy instead.
/// Entry rules: https://docs.vulkan.org/spec/latest/chapters/raytraversal.html#ray-intersection-candidate-determination
/// Chained caps: https://docs.vulkan.org/refpages/latest/refpages/source/VkRayTracingLssPrimitiveEndCapsModeNV.html
/// Coincident centers intersect only the larger sphere, including exit hits from inside.
/// With no spherical joints or with both radii zero, coincident centers produce no hit.
/// @param ray_origin Ray origin position.
/// @param ray_dir Unit ray direction vector.
/// @param sphere_a Sphere (3D position + radius) at one end point.
/// @param sphere_b Sphere at the other end point.
/// @param use_sphere_joints Indicating whether we test ray-sphere intersection at curve joints or not.
/// @param[out] result The closest intersection distance t, and parameter u for linear interpolation (between 0 and 1).
/// @return True if the ray intersects the linear swept sphere segment.
public bool intersect_ray_linear_swept_sphere_han19(
    float3 ray_origin,
    const float3 ray_dir,
    const float4 sphere_a,
    const float4 sphere_b,
    const bool use_sphere_joints,
    out float2 result
)
{
    result = float2(float.MAX);

    if (all(sphere_a.xyz == sphere_b.xyz)) {
        const float radius = max(sphere_a.w, sphere_b.w);
        float t;
        if (!use_sphere_joints || radius == 0.0 || !intersect_ray_sphere(ray_origin, ray_dir, sphere_a.xyz, radius, t))
            return false;
        // Keep the intersection helper's endpoint parameter convention; ties select A.
        result = float2(t, sphere_b.w > sphere_a.w ? 1.0 : 0.0);
        return true;
    }

    const bool reversed = sphere_a.w > sphere_b.w;
    const float3 c1 = select(reversed, sphere_b.xyz, sphere_a.xyz);
    const float3 c2 = select(reversed, sphere_a.xyz, sphere_b.xyz);
    const float r1 = select(reversed, sphere_b.w, sphere_a.w);
    const float r2 = select(reversed, sphere_a.w, sphere_b.w);

    // Perpendicular distance to closest of (p0, p1) minus the max radius.
    float t0 = min(dot(ray_dir, c1 - ray_origin), dot(ray_dir, c2 - ray_origin)) - max(r1, r2);
    t0 = max(0.0, t0);
    // For better numerical stability, push the ray to be as close as possible to the curve.
    ray_origin += t0 * ray_dir;

    if (use_sphere_joints) {
        // Intersecting two sphere endcaps.
        float t;
        if (intersect_ray_sphere(ray_origin, ray_dir, c1, r1, t)) {
            if (t < result.x) {
                result.x = t;
                result.y = (reversed ? 1.0 : 0.0);
            }
        }
        if (intersect_ray_sphere(ray_origin, ray_dir, c2, r2, t)) {
            if (t < result.x) {
                result.x = t;
                result.y = (reversed ? 0.0 : 1.0);
            }
        }
    }

    float3 C = c2 - c1;
    const float length_C = length(C);
    // Equal radii form a cylinder; the cone-apex parameterization below is singular in this case.
    if (r1 == r2) {
        if (length_C > 0.0) {
            const float3 axis = C / length_C;
            float t;
            if (intersect_ray_open_cylinder(
                    ray_origin,
                    ray_dir,
                    0.5 * (c1 + c2),
                    axis,
                    r1,
                    0.5 * length_C,
                    t,
                    0.0,
                    result.x
                )) {
                result.x = t;
                result.y = saturate(dot(ray_origin + t * ray_dir - c1, axis) / length_C);
            }
        }
        result.x += t0;
        return result.x < float.MAX;
    }

    // Intersecting cone.
    C /= length_C;

    const float p1 = length_C * r1 / (r2 - r1);
    const float p2 = p1 * r2 / r1;
    const float3 A = c1 - p1 * C;
    const float z1 = p1 - r1 * r1 / p1;
    const float z2 = p2 - r2 * r2 / p2;
    const float w = p2 * r2 / sqrt(p2 * p2 - r2 * r2);

    const float3 vz = C;
    float3 vx, vy;
    branchless_ONB(vz, vx, vy);

    const float tmp1 = 1.0 / z2;
    const float tmp2 = p2 / w;

    // Row-major matrix.
    float4x4 M = {
        tmp1 * tmp2 * float4(vx, -dot(vx, A)), // row 0
        tmp1 * tmp2 * float4(vy, -dot(vy, A)), // row 1
        tmp1 * float4(vz, -dot(vz, A)),        // row 2
        float4(0, 0, 0, 1)                     // row 3
    };

    const float z_cap = z1 * tmp1;
    const float3 ray_origin_l = mul(M, float4(ray_origin, 1.0)).xyz;
    const float3 ray_dir_l = mul((float3x3)M, ray_dir);

    const float a = ray_dir_l.x * ray_dir_l.x + ray_dir_l.y * ray_dir_l.y - ray_dir_l.z * ray_dir_l.z;
    const float b = 2.0 * (ray_origin_l.x * ray_dir_l.x + ray_origin_l.y * ray_dir_l.y - ray_origin_l.z * ray_dir_l.z);
    const float c = ray_origin_l.x * ray_origin_l.x + ray_origin_l.y * ray_origin_l.y - ray_origin_l.z * ray_origin_l.z;
    const float disc = b * b - 4 * a * c;
    if (disc >= 0) {
        const float sqrt_disc = sqrt(disc);
        const float inv2a = 0.5 / a;

        [unroll]
        for (int i = 0; i < 2; i++) {
            float t = (-b + (i * 2.0 - 1.0) * sqrt_disc) * inv2a;
            if (t >= 0 && t < result.x) {
                // Check if z is in the valid range.
                const float z = ray_origin_l.z + t * ray_dir_l.z;
                if (z >= z_cap && z <= 1.0) {
                    const float u = (z - z_cap) / (1.0 - z_cap);
                    result.x = t;
                    result.y = (reversed ? 1.0 - u : u);
                }
            }
        }
    }

    result.x += t0;
    return (result.x < float.MAX);
}
