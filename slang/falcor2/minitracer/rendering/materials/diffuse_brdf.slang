// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing rendering;

/// Simple lambert diffuse BRDF.
public struct DiffuseBRDF : IDifferentiable
{
    float3 diffuse; ///< Diffuse albedo.

    [Differentiable]
    public __init(float3 albedo) { this.diffuse = albedo; }

    [Differentiable]
    public float3 eval(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < MIN_COS_THETA)
            return float3(0.f);

        float3 fd = diffuse_eval(wi, wo, diffuse);
        return fd * wo.z;
    }

    [Differentiable]
    public bool
    sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, inout UniformSampleGenerator sg)
    {
        wo = {};
        pdf = {};
        weight = {};

        float2 u = no_diff next_float2(sg);

        wo = square_to_cosine_hemisphere_concentric(u);
        pdf = square_to_cosine_hemisphere_concentric_pdf(wo);

        if (min(wi.z, wo.z) < MIN_COS_THETA || pdf == 0.f)
            return false;

        weight = eval(wi, wo) / pdf;
        return true;
    }

    [Differentiable]
    public float eval_pdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < MIN_COS_THETA)
            return 0.f;

        return diffuse_pdf(wo);
    }

    // Internal functions

    [Differentiable]
    float3 diffuse_eval(const float3 wi, const float3 wo, const float3 albedo) { return float.INV_PI * albedo; }

    [Differentiable]
    float diffuse_pdf(const float3 wo) { return max(0.f, wo.z) * float.INV_PI; }
};
