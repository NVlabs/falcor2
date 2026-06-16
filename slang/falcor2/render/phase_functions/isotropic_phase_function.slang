// SPDX-License-Identifier: Apache-2.0

implementing render;


/// Isotropic phase function.
/// f_p(wi, wo) = 1 / 4PI
public struct IsotropicPhaseFunction : IPhaseFunction
{
    public __init() { }

    public float3 eval(const float3 wi, const float3 wo) { return float.INV_FOUR_PI; }

    public bool
    sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, inout S sg)
    {
        wo = square_to_uniform_sphere(next_float2(sg));
        pdf = float.INV_FOUR_PI;
        weight = 1.f;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo) { return float.INV_FOUR_PI; }
};
