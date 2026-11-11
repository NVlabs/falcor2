// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

public struct Frame : IDifferentiable
{
    // Frame uses the common shading-space convention where the surface normal is local +Z.
    // This is independent of the engine's world-space up axis, which may be Y-up.
    public float3 normal = {};
    public float3 tangent = {};
    public float3 bitangent = {};

    public enum Component {
        normal,
        tangent,
    }

    public __init() { }

    public static Frame identity()
    {
        Frame frame = {};
        frame.normal = float3(0.0, 0.0, 1.0);
        frame.tangent = float3(1.0, 0.0, 0.0);
        frame.bitangent = float3(0.0, 1.0, 0.0);
        return frame;
    }

    [Differentiable]
    public __init(float3 normal)
    {
        // Using branchlessONB from
        // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
        this.normal = normal;
        branchless_ONB(this.normal, this.tangent, this.bitangent);
    }

    // normal and tangent are both assumed to be normalized, but not necessarily orthonormal.
    // handedness is with respect to the UV tangent space, not the geometry orientation.
    [Differentiable]
    public __init(
        float3 normal,
        float3 tangent,
        float handedness = 1.0,
        Component preserved_component = Component::normal
    )
    {
        const float3 unnormalized_bitangent = cross(normal, tangent);
        // Collinear inputs do not define a tangent plane. Keep the requested axis
        // and choose a stable perpendicular pair instead of normalizing zero.
        if (dot(unnormalized_bitangent, unnormalized_bitangent) <= 1.0e-12) {
            if (preserved_component == Component::normal) {
                this.normal = normal;
                branchless_ONB(normal, this.tangent, this.bitangent);
            } else {
                this.tangent = tangent;
                branchless_ONB(tangent, this.bitangent, this.normal);
            }
        } else {
            this.bitangent = normalize(unnormalized_bitangent);
            if (preserved_component == Component::normal) {
                this.normal = normal;
                this.tangent = cross(bitangent, normal);
            } else // preserve tangent, used for curves
            {
                this.tangent = tangent;
                this.normal = cross(tangent, bitangent);
            }
        }
        this.bitangent *= handedness;
    }

    [Differentiable]
    public float3 to_local(float3 w) { return float3(dot(tangent, w), dot(bitangent, w), dot(normal, w)); }

    [Differentiable]
    public float3 to_global(float3 w) { return w.x * tangent + w.y * bitangent + w.z * normal; }

    // Returns sign that gets applied to such that bitangent = cross(normal, tangent) * handedness_sign();
    // Returns +1 for right handed, and -1 for left handed frames.
    public float handedness_sign() { return dot(cross(normal, tangent), bitangent) >= 0.f ? 1.f : -1.f; }
};

/// Given v1 and v2 where v1 is normalized and v1 and v2 define a plane
/// (v2 is not zero length and not colinear to v1), this method adjusts v2,
/// such that it is orthonormal to v1, lying in the plane defined by the two vectors.
public void orthonormalize(const float3 v1, inout float3 v2)
{
    v2 = normalize(v2 - dot(v1, v2) * v1);
}

public void branchless_ONB(float3 normal, out float3 tangent, out float3 bitangent)
{
    // Using branchlessONB from
    // https://graphics.pixar.com/library/OrthonormalB/paper.pdf

    // Workaround because slang's copysign is currently not differentiable
    // https://github.com/shader-slang/slang/issues/7549
    // float sign = copysign(1.0, normal.z);
    float sign = select(normal.z < 0.0, -1.0, 1.0);
    const float a = -1.0f / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    tangent = float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    bitangent = float3(b, sign + normal.y * normal.y * a, -normal.y);
}

float3 adjust_shading_normal_impl(
    const float3 shading_normal,
    const float3 geometric_normal,
    const float3 view_dir,
    out bool was_adjusted
)
{
    was_adjusted = false;

    // The shading normal Ns may lie on either side depending on whether we're shading the front or back.
    // We orient Ns to lie on the same side as the view direction and Ng for the computations below.
    // The final adjusted normal is oriented to lie on the same side as the original shading normal.
    float3 Ng = geometric_normal;
    float sign_N = dot(shading_normal, Ng) >= 0.f ? 1.f : -1.f;
    float3 Ns = sign_N * shading_normal;

    // Algorithm from Appendix A.3 of https://arxiv.org/abs/1705.01263
    // The specular reflection of the view direction is computed.
    // If the reflection vector lies under the horizon, the shading normal is adjusted.
    // This ensures that the majority of the reflection lobe lies on the same side,
    // but it does not guarantee that all directions are valid in the general case.

    // Specular reflection in shading normal
    float3 R = reflect(-view_dir, Ns);
    float a = dot(Ng, R);
    if (a < 0) // Perturb normal
    {
        float b = max(0.001, dot(Ns, Ng));
        was_adjusted = true;
        return normalize(sign_N * (view_dir + normalize(R - Ns * a / b)));
    }
    return shading_normal;
}

/// Adjusts a unit-length shading normal to avoid invalid reflection directions.
/// The geometric normal and view direction must be unit length and lie in the same hemisphere.
/// The result remains on the shading normal's authored side of the geometric normal.
public float3 adjust_shading_normal(const float3 shading_normal, const float3 geometric_normal, const float3 view_dir)
{
    bool was_adjusted;
    return adjust_shading_normal_impl(shading_normal, geometric_normal, view_dir, was_adjusted);
}

public void adjust_shading_normal(inout Frame sf, const float3 geometric_normal, const float3 view_dir)
{
    bool was_adjusted;
    const float3 adjusted_normal = adjust_shading_normal_impl(sf.normal, geometric_normal, view_dir, was_adjusted);
    if (was_adjusted)
        sf = Frame(adjusted_normal, sf.tangent, sf.handedness_sign());
}

/// Repairs a shading frame into the hemisphere of oriented_geometric_normal, preserving handedness.
/// All normals and view_dir must be unit length. front_geometric_normal must face view_dir;
/// oriented_geometric_normal selects the intended shading side and may face away for transmission.
public void adjust_shading_frame(
    inout Frame sf,
    const float3 oriented_geometric_normal,
    const float3 front_geometric_normal,
    const float3 view_dir
)
{
    const float cos_theta = dot(sf.normal, oriented_geometric_normal);
    if (cos_theta < 0.0) {
        // A mapped normal can cross the geometric horizon. Keep its tangential direction,
        // but put it on the intended side before the side-preserving reflection repair.
        const float3 normal
            = normalize(sf.normal - cos_theta * oriented_geometric_normal + 0.001 * oriented_geometric_normal);
        sf = Frame(normal, sf.tangent, sf.handedness_sign());
    }
    adjust_shading_normal(sf, front_geometric_normal, view_dir);
}
