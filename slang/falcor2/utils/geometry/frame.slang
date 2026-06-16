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
    [Differentiable]
    public __init(float3 normal, float3 tangent, Component preserved_component = Component::normal)
    {
        // Only useful for extreme bump cases where tangent and normal can become almost colinear.
        // if (abs(dot(normal, tangent)) >= 0.999) {
        //     this.normal = normal;
        //     branchless_ONB(this.normal, this.tangent, this.bitangent);
        //     return;
        // }
        if (preserved_component == Component::normal) {
            this.normal = normal;
            this.bitangent = normalize(cross(normal, tangent));
            this.tangent = cross(bitangent, normal);
        } else // preserve tangent, used for curves
        {
            this.tangent = tangent;
            this.bitangent = normalize(cross(normal, tangent));
            this.normal = cross(tangent, bitangent);
        }
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

public void adjust_shading_normal(inout Frame sf, const float3 geometric_normal, const float3 view_dir)
{
    // Note: sd.V and Ng below always lie on the same side (as the front-facing flag is computed based on sd.V).
    // The shading normal sf.N may lie on either side depending on whether we're shading the front or back.
    // We orient Ns to lie on the same side as sd.V and Ng for the computations below.
    // The final adjusted normal is oriented to lie on the same side as the original shading normal.
    float3 Ng = geometric_normal;
    float sign_N = dot(sf.normal, Ng) >= 0.f ? 1.f : -1.f;
    float3 Ns = sign_N * sf.normal;

    // Blend the shading normal towards the geometric normal at grazing angles.
    // This is to avoid the view vector from becoming back-facing.
    const float COS_THETA_THRESHOLD = 0.1f;
    float cos_theta = dot(view_dir, Ns);
    if (cos_theta <= COS_THETA_THRESHOLD) {
        float t = saturate(cos_theta * (1.f / COS_THETA_THRESHOLD));
        sf = Frame(normalize(sign_N * lerp(Ng, Ns, t)), sf.tangent);
    }
}
