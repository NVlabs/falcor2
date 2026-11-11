// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off: replacement blocks below are not standalone Slang syntax.
public struct $MxMaterialInstanceName : IMaterialInstance
{
    typealias mtlx_ns = $MxAliasesName;
    public Frame shading_frame_ws;
    public mtlx_ns::RootBSDF bsdf;
    public float3 emissive_radiance = {};
    public BSDFEval eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both, S : ISampleGenerator>(
        const SurfaceInteraction si,
        const float3 wo_ws,
        inout S sg
    )
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);
        return bsdf.eval_with_pdf(wi_ls, wo_ls, sg, BSDFContext());
    }
    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out MaterialSample sample)
    {
        sample = {};
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        BSDFSample bsdf_sample;
        bool valid = bsdf.sample(wi_ls, bsdf_sample, sg, BSDFContext());
        sample.wo_ws = shading_frame_ws.to_global(bsdf_sample.wo);
        sample.pdf = bsdf_sample.pdf;
        sample.weight = bsdf_sample.weight;
        sample.flags = bsdf_sample.flags;
        return valid;
    }
    public override bool is_emissive() { return any(emissive_radiance > 0.0); }
    public override float3 eval_emissive_radiance() { return emissive_radiance; }
    $MxCollectPropertiesMethod$MxCollectExtraBsdfPropertiesMethod public BSDFFlags
    get_lobe_types(const SurfaceInteraction si)
    {
        return $MxLobes;
    }
};

// clang-format on
