// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

__exported import scattering_tests;

/// Known integral of one, with half of all attempts rejected. Multiplying the
/// claimed density leaves eval/pdf consistent but breaks distribution and mean.
public struct RejectionTarget : scattering_tests::IScatteringTestTarget, chi2::ITarget
{
    public float pdf_scale;
    public typealias Domain = chi2::SphericalDomain;
    public Domain get_domain() { return {}; }

    public bool sample(BSDFSampleInput input, out BSDFSample result)
    {
        result = {};
        if (input.u_lobe < 0.5)
            return false;
        const float z = input.u_dir.x;
        const float phi = 2.0 * float.PI * input.u_dir.y;
        const float r = sqrt(1.0 - z * z);
        result.wo = float3(r * cos(phi), r * sin(phi), z);
        result.pdf = pdf_scale / (4.0 * float.PI);
        result.weight = float3(2.0 / pdf_scale);
        result.flags = BSDFFlags::glossy_reflection;
        return true;
    }

    public bool sample<SG : ISampleGenerator>(inout SG sg, out float3 p)
    {
        BSDFSample result = {};
        const bool valid = sample(make_bsdf_sample_input(sg), result);
        p = result.wo;
        return valid;
    }


    public float pdf(float3 wo) { return eval_with_pdf(wo).pdf; }

    public BSDFEval eval_with_pdf(float3 wo, BSDFComponent component = BSDFComponent::full_response)
    {
        return component == BSDFComponent::full_response && wo.z >= 0.0
            ? BSDFEval(float3(1.0 / (2.0 * float.PI)), pdf_scale / (4.0 * float.PI))
            : BSDFEval();
    }
};

/// Two overlapping uniform-hemisphere components with different physical and proposal weights.
/// Independent corruption controls distinguish contribution identity, joint density, marginal density and weight.
public struct ComponentTarget : scattering_tests::IScatteringTestTarget
{
    public bool full_response;
    public bool claim_full_response;
    public uint component_xor;
    public float joint_pdf_scale;
    public float marginal_pdf_scale;
    public float weight_scale;

    public bool sample(BSDFSampleInput input, out BSDFSample result)
    {
        result = {};
        const float z = input.u_dir.x;
        const float phi = 2.0 * float.PI * input.u_dir.y;
        const float r = sqrt(1.0 - z * z);
        result.wo = float3(r * cos(phi), r * sin(phi), z);
        const uint component = input.u_lobe < 0.25 ? 0 : 1;
        const float proposal_weight = component == 0 ? 0.25 : 0.75;
        const float physical_weight = component == 0 ? 0.2 : 0.8;
        result.weight = float3((full_response ? 1.0 : physical_weight / proposal_weight) * weight_scale);
        result.pdf = marginal_pdf_scale / (2.0 * float.PI);
        result.component = full_response || claim_full_response ? BSDFComponent::full_response
                                                                : BSDFComponent(component ^ component_xor);
        result.flags = BSDFFlags::glossy_reflection;
        return true;
    }

    public BSDFEval eval_with_pdf(float3 wo, BSDFComponent component = BSDFComponent::full_response)
    {
        if (wo.z < 0.0)
            return BSDFEval();
        const float pdf = 1.0 / (2.0 * float.PI);
        if (component == BSDFComponent::full_response)
            return BSDFEval(float3(pdf), pdf);
        if (uint(component) > 1)
            return BSDFEval();
        const float proposal_weight = uint(component) == 0 ? 0.25 : 0.75;
        const float physical_weight = uint(component) == 0 ? 0.2 : 0.8;
        return BSDFEval(float3(physical_weight * pdf), proposal_weight * pdf * joint_pdf_scale);
    }
};
