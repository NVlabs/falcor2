// SPDX-License-Identifier: Apache-2.0

import slangpy;
__exported import falcor2.utils;

public struct SquareToCosineHemispherePolarTarget : chi2::ITarget
{
    // TODO: This is a workaround for slang issue
    // https://github.com/shader-slang/slang/issues/8125
#ifdef __TARGET_CUDA__
    int dummy;
#endif
    typedef chi2::SphericalDomain Domain;
    public Domain get_domain() { return chi2::SphericalDomain(); }
    public bool sample<SG : ISampleGenerator>(inout SG sg, out float3 p)
    {
        p = square_to_cosine_hemisphere_polar(next_float2(sg));
        return true;
    }
    public float pdf(float3 p) { return square_to_cosine_hemisphere_polar_pdf(p, true); }
};

public struct SquareToCosineHemisphereConcentricTarget : chi2::ITarget
{
    // TODO: This is a workaround for slang issue
    // https://github.com/shader-slang/slang/issues/8125
#ifdef __TARGET_CUDA__
    int dummy;
#endif
    typedef chi2::SphericalDomain Domain;
    public Domain get_domain() { return chi2::SphericalDomain(); }
    public bool sample<SG : ISampleGenerator>(inout SG sg, out float3 p)
    {
        p = square_to_cosine_hemisphere_concentric(next_float2(sg));
        return true;
    }
    public float pdf(float3 p) { return square_to_cosine_hemisphere_concentric_pdf(p, true); }
};
