// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

public float3 density(float alpha, float3 h)
{
    return float3(evalNdfGGX(alpha, h), evalNdfGGX(alpha, h.z), evalPdfGGX_VNDF(alpha, float3(0.0, 0.0, 1.0), h));
}

public float4 reflection(float alpha, float3 wo)
{
    SpecularMicrofacetBRDF bsdf = SpecularMicrofacetBRDF(float3(1.0), alpha, BSDFFlags::all);
    UniformSampleGenerator sg = UniformSampleGenerator(0);
    BSDFEval result = bsdf.eval_with_pdf(float3(0.0, 0.0, 1.0), wo, sg, BSDFContext());
    return float4(result.value, result.pdf);
}

public float3 half_vector(float3 wo)
{
    return normalize(float3(0.0, 0.0, 1.0) + wo);
}
