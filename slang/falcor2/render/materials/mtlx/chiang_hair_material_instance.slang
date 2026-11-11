// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mtlx;

import falcor2.render;
import falcor2.utils;

namespace mtlx {

/// A Chiang closure integrates transport through a fiber in air over the full sphere.
/// Its opposite-hemisphere lobes do not cross a tracked surface medium.
public struct MXChiangHairMaterialInstance : IMaterialInstance
{
    public Frame shading_frame_ws;
    public MxChiangHairBSDF bsdf;
    public float3 emissive_radiance = {};

    public BSDFEval eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both, S : ISampleGenerator>(
        const SurfaceInteraction si,
        const float3 wo_ws,
        inout S sg
    )
    {
        return bsdf
            .eval_with_pdf(shading_frame_ws.to_local(si.wi_ws), shading_frame_ws.to_local(wo_ws), sg, BSDFContext());
    }

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out MaterialSample sample)
    {
        sample = {};
        BSDFSample bsdf_sample;
        if (!bsdf.sample(shading_frame_ws.to_local(si.wi_ws), bsdf_sample, sg, BSDFContext()))
            return false;
        sample.wo_ws = shading_frame_ws.to_global(bsdf_sample.wo);
        sample.pdf = bsdf_sample.pdf;
        sample.weight = bsdf_sample.weight;
        sample.flags = bsdf_sample.flags;
        return true;
    }

    public override bool is_emissive() { return any(emissive_radiance > 0.0); }
    public override float3 eval_emissive_radiance() { return emissive_radiance; }
    public BSDFFlags get_lobe_types(const SurfaceInteraction si) { return BSDFFlags::glossy_curve; }

    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        const float3 wi = shading_frame_ws.to_local(si.wi_ws);
        const float3 albedo = bsdf.directional_albedo(wi);
        const float2 roughness = bsdf.eval_roughness(wi).roughness;
        const float diffuse_fraction = min(roughness.x + roughness.y, 1.0);
        MaterialProperties result = {};
        // Retain Falcor's roughness-based split for denoiser guides; these are not separate hair lobes.
        result.diffuse_reflection_albedo = albedo * diffuse_fraction;
        result.specular_reflection_albedo = albedo * (1.0 - diffuse_fraction);
        result.specular_reflectance = result.specular_reflection_albedo;
        result.material_color = saturate(result.diffuse_reflection_albedo);
        result.roughness = (roughness.x + roughness.y) * 0.5;
        result.guide_normal = shading_frame_ws.normal;
        result.emission = emissive_radiance;
        return result;
    }

    public override Optional<ExtraBsdfProperties>
    collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        const float3 wi = shading_frame_ws.to_local(si.wi_ws);
        ExtraBsdfProperties result = {};
        result.bsdf_count = 1;
        result.bsdf_N[0] = si.shading_frame_ws.to_local(shading_frame_ws.normal);
        result.bsdf_T[0] = si.shading_frame_ws.to_local(shading_frame_ws.tangent);
        result.bsdf_B[0] = si.shading_frame_ws.to_local(shading_frame_ws.bitangent);
        result.bsdf_albedo[0] = bsdf.directional_albedo(wi);
        result.bsdf_weight[0] = float3(1.0);
        result.bsdf_roughness[0] = bsdf.eval_roughness(wi).roughness;
        result.bsdf_scratch[0] = float3(0.0);
        return result;
    }
}

} // namespace mtlx
