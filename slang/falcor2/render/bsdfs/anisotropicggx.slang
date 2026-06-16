// SPDX-License-Identifier: Apache-2.0

implementing render;

/// This class implements sampling and evaluation of an anisotropic GGX.
/// The class uses the visible distribution of normals for sampling.
public typealias AnisotropicGGX = AnisotropicGGXPolicy<1000, false>;

/// Policy-specialized implementation of AnisotropicGGX.
///
/// TSingularCutoffInverse controls the alpha threshold below which the lobe is
/// considered singular. The value is expressed as an inverse integer because
/// Slang generic value parameters do not currently support floats.
///
/// TClampBelowCutoff switches the policy from "treat below-threshold lobes as
/// singular" to "clamp alpha up to the threshold and keep evaluating finite
/// GGX". The AnisotropicGGX alias preserves the historical behavior, while
/// specific material systems can opt into finite near-delta evaluation when
/// their reference implementation does not use a singular branch.
public struct AnisotropicGGXPolicy<let TSingularCutoffInverse : int, let TClampBelowCutoff : bool> : IDifferentiable
{
    public float2 alpha = {};

    [Differentiable]
    public __init()
    {
        float singular_cutoff = 1.0 / float(TSingularCutoffInverse);
        this.alpha = TClampBelowCutoff ? float2(singular_cutoff) : float2(0.0);
    }

    [Differentiable]
    public __init(float2 alpha)
    {
        float singular_cutoff = 1.0 / float(TSingularCutoffInverse);
        this.alpha = TClampBelowCutoff ? max(alpha, float2(singular_cutoff)) : alpha;
    }

    /// A 'singular' distribution is one which would be more ideally treated as an
    /// ideal mirror. It can't be sampled reliably.
    public bool is_singular() { return !TClampBelowCutoff && all(alpha < 1.0 / float(TSingularCutoffInverse)); }

    [Differentiable]
    public float isotropic_roughness() { return max(alpha.x, alpha.y); }

    /// Computes the Smith lambda function. This function is symmetric with respect to direction,
    /// i.e. smith_lambda(v) = smith_lambda(-v). The result returned is equivalent to assuming v.z > 0.
    [Differentiable]
    public float smith_lambda(const float3 v)
    {
        float cos_theta_sq = v.z * v.z;
        float sin_theta_sq = max(1.0 - cos_theta_sq, 0.0);
        float tan_theta_sq = sin_theta_sq / cos_theta_sq;
        if (isinf(tan_theta_sq))
            return 0.0;

        float2 cos_sin_phi_sq = sin_theta_sq == 0.0 ? float2(1.0, 0.0) : float2(v.x * v.x, v.y * v.y) / sin_theta_sq;
        float alpha_sq = dot(cos_sin_phi_sq, alpha * alpha);
        return (sqrt(1.0 + alpha_sq * tan_theta_sq) - 1.0) * 0.5;
    }

    /// Compute the monostatic Smith shadowing function. This function returns the result
    /// assuming w.z > 0. This is what you want for single scattering, but you need to be careful
    /// for multiple scattering on microfacets.
    [Differentiable]
    public float G1(const float3 w) { return 1.0 / (1.0 + smith_lambda(w)); }

    /// Compute the bistatic Smith shadowing function for reflection.
    /// This will be off for refraction, where we would have to use the beta
    /// function instead. This is not worth the trouble unless we do
    /// multiple scattering on microfacet dielectrics as well.
    [Differentiable]
    public float G2(const float3 wo, const float3 wi) { return 1.0 / (1.0 + smith_lambda(wo) + smith_lambda(wi)); }

    /// Computes the normal distribution function, i.e. D(h)
    [Differentiable]
    public float eval_ndf(const float3 h)
    {
        float cos_theta_sq = h.z * h.z;
        float sin_theta_sq = max(1.0 - cos_theta_sq, 0.0);
        float tan_theta_sq = sin_theta_sq / cos_theta_sq;
        if (isinf(tan_theta_sq))
            return 0.0;

        float cos_theta_qu = cos_theta_sq * cos_theta_sq;
        if (cos_theta_qu < 1e-16)
            return 0.0;

        float2 cos_sin_phi_sq = sin_theta_sq == 0.0 ? float2(1.0, 0.0) : float2(h.x * h.x, h.y * h.y) / sin_theta_sq;
        float2 alpha_sq = cos_sin_phi_sq / (alpha * alpha);
        float e = tan_theta_sq * (alpha_sq.x + alpha_sq.y);
        return 1.0 / (float.PI* alpha.x* alpha.y* cos_theta_qu* sqr(1.0 + e));
    }

    /// Computes the distribution of visible normals, i.e. D_wi(h)
    [Differentiable]
    public float eval_vndf(const float3 w, const float3 h)
    {
        return w.z == 0.0 ? 0.0 : (G1(w) * abs(dot(w, h)) * eval_ndf(h)) / abs(w.z);
    }

    /// Samples the distribution of normals visible from direction w.
    public float3 sample<S : ISampleGenerator>(const float3 w, inout S sg)
    {
        float3 wh = normalize(float3(alpha.x * w.x, alpha.y * w.y, w.z));
        if (wh.z < 0.0)
            wh = -wh;

        float3 T1 = dot(wh.xy, wh.xy) > 0.0 ? normalize(float3(-wh.y, wh.x, 0.0)) : float3(1.0, 0.0, 0.0);
        float3 T2 = cross(wh, T1);

        float r = sqrt(next_float(sg));
        float phi = 2.0 * float.PI * next_float(sg);
        float t1 = r * cos(phi);
        float t2 = r * sin(phi);
        float s = 0.5 * (1.0 + wh.z);
        t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

        float3 nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * wh;
        return normalize(float3(alpha.x * nh.x, alpha.y * nh.y, max(0.0, nh.z)));
    }

    /// Returns the PDF of sampling normal h from VNDF(w)
    float eval_pdf(float3 w, float3 h) { return eval_vndf(w, h); }
};
