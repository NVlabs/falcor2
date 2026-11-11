// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/testing/shader_test.slangh"
import falcor2.render;
import falcor2.render.materials.mtlx.mtlx;

TEST_CASE(materialx_conductor_albedo_preserves_reflective_channels)
{
    // artistic_ior reflectivity (0.9, 0.04, 0), edge_color (0.75, 0, 0).
    // The old average-Fresnel polynomial incorrectly zeroed both reflective channels.
    const float3 expected = float3(0.9, 0.04, 0.0);
    mtlx::MxConductorFresnel fresnel
        = mtlx::MxConductorFresnel(float3(9.5328903, 1.5, 1.0), float3(16.4430485, 0.0, 0.0));
    CHECK_APPROX(fresnel.eval(1.0), expected, 1e-6);

    for (int i = 0; i < 3; ++i) {
        const float cosine = i == 0 ? 1.0 : (i == 1 ? 0.5 : 0.1);
        const float3 wi = float3(sqrt(1.0 - cosine * cosine), 0.0, cosine);
        mtlx::MxConductorTurquinAnalyticCompensationBSDF bsdf
            = mtlx::MxConductorTurquinAnalyticCompensationBSDF(float2(0.05), fresnel, false);
        AlbedoContributions albedo = bsdf.eval_albedo(wi);
        CHECK_APPROX(bsdf.directional_albedo(wi), expected, 1e-6);
        CHECK_APPROX(albedo.reflection, expected, 1e-6);
        CHECK_APPROX(albedo.absorption_r, float3(1.0) - expected, 1e-6);
        CHECK_EQ(albedo.transmission, float3(0.0));
        CHECK_EQ(albedo.absorption_t, float3(0.0));
        CHECK_EQ(bsdf.layer_opacity(wi), float3(1.0));
    }
}

TEST_CASE(materialx_conductor_zero_thickness_airy_uses_f0_estimate)
{
    // A white-edge artistic metal must retain its authored 90% facing reflectivity.
    mtlx::MxConductorAiryFresnel fresnel
        = mtlx::MxConductorAiryFresnel(float3(0.052631579), float3(0.998613998), 0.0, 1.5);
    CHECK_APPROX(fresnel.directional_albedo(1.0, 0.05), float3(0.9), 1e-6);
    CHECK_APPROX(fresnel.directional_albedo(0.1, 0.75), float3(0.9), 1e-6);
}
