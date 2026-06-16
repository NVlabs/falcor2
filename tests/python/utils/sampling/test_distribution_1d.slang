// SPDX-License-Identifier: Apache-2.0

import slangpy;
__exported import falcor2.utils;

public struct DiscreteDistribution1DTarget : chi2::ITarget
{
    public DiscreteDistribution1D dist;

    typedef chi2::DiscreteLineDomain Domain;
    public Domain get_domain() { return chi2::DiscreteLineDomain(0, dist.size); }
    public bool sample<SG : ISampleGenerator>(inout SG sg, out float1 p)
    {
        uint index = {};
        float pdf = {};
        bool result = dist.sample(next_float(sg), index, pdf);
        p = float1(index);
        return result;
    }
    public float pdf(float1 p) { return dist.eval_pdf(uint(p.x)); }
};

public struct AliasTable1DTarget : chi2::ITarget
{
    public AliasTable1D dist;

    typedef chi2::DiscreteLineDomain Domain;
    public Domain get_domain() { return chi2::DiscreteLineDomain(0, dist.size); }
    public bool sample<SG : ISampleGenerator>(inout SG sg, out float1 p)
    {
        uint index = {};
        float pdf = {};
        bool result = dist.sample(next_float(sg), index, pdf);
        p = float1(index);
        return result;
    }
    public float pdf(float1 p) { return dist.eval_pdf(uint(p.x)); }
};
