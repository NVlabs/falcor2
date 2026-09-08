// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define MATERIALX_HAS_EMISSION $MxHasEmission
#define MATERIALX_EVALUATE_OPACITY $MxEvaluateOpacity
#define MATERIALX_OPACITY_IS_OPAQUE $MxOpacityIsOpaque
#define MATERIALX_STATIC_OPACITY $MxStaticOpacity

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

public struct $MxMaterialName : IMaterial
{
    typealias MaterialInstance = $MxMaterialInstanceName;
    typealias mtlx_ns = $MxAliasesName;
    public MaterialHeader header;
#if MATERIALX_PAYLOAD_SIZE <= MATERIAL_PAYLOAD_LIMIT
    public $MxMaterialDataName data;
#else
    public BufferHandle data_buffer;
#endif
    static SurfaceInteraction prepare_graph_si(const SurfaceInteraction si, const MaterialInstanceHints hints)
    {
        SurfaceInteraction graph_si = si;
        if (!graph_si.front_facing)
            graph_si.shading_frame_ws.normal = -graph_si.shading_frame_ws.normal;
        if (hints & MaterialInstanceHints::adjust_shading_normal) {
            adjust_shading_normal(graph_si.shading_frame_ws, graph_si.front_normal_ws(), graph_si.wi_ws);
        }
        return graph_si;
    }

    static MaterialInstance setup_material_instance_impl<TLodSampler : ILodSampler>(
        $MxMaterialDataName data,
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints,
        out float3 opacity_weighted_emission
    )
    {
        typealias Graph = $MxGraphName<TLodSampler>;
        MaterialInstance result = {};
        SurfaceInteraction graph_si = prepare_graph_si(si, hints);
        result.shading_frame_ws = graph_si.shading_frame_ws;
        let mtlx_graph = Graph(data, graph_si, lod_sampler, uint(hints));
        let stack = mtlx_graph.mx_stack;
        $MxRootValueLines result.bsdf = $MxRootValueExpr;
#if MATERIALX_HAS_EMISSION
        opacity_weighted_emission = $MxGraphEmission;
#if MATERIALX_OPACITY_IS_OPAQUE
        result.emissive_radiance = opacity_weighted_emission;
#else
#if MATERIALX_EVALUATE_OPACITY
        let graph_opacity = $MxGraphOpacity;
#else
        let graph_opacity = MATERIALX_STATIC_OPACITY;
#endif
        result.emissive_radiance = graph_opacity > 0.0 ? opacity_weighted_emission / graph_opacity : float3(0.0);
#endif
#else
        opacity_weighted_emission = float3(0.0);
#endif
        return result;
    }
    MaterialInstance setup_material_instance<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints
    )
    {
#if MATERIALX_PAYLOAD_SIZE > MATERIAL_PAYLOAD_LIMIT
        let data = data_buffer.load<$MxMaterialDataName>(0);
#endif
        float3 opacity_weighted_emission;
        return setup_material_instance_impl(data, si, lod_sampler, hints, opacity_weighted_emission);
    }
    public override float eval_opacity<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
#if MATERIALX_EVALUATE_OPACITY
#if MATERIALX_PAYLOAD_SIZE > MATERIAL_PAYLOAD_LIMIT
        let data = data_buffer.load<$MxMaterialDataName>(0);
#endif
        typealias Graph = $MxGraphName<TLodSampler>;
        SurfaceInteraction graph_si = prepare_graph_si(si, hints);
        let mtlx_graph = Graph(data, graph_si, lod_sampler, uint(hints));
        return $MxGraphOpacity;
#else
        return MATERIALX_STATIC_OPACITY;
#endif
    }
    public override bool is_emissive() { return MATERIALX_HAS_EMISSION != 0; }
    public override float3 eval_emissive_radiance<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
#if MATERIALX_HAS_EMISSION
#if MATERIALX_PAYLOAD_SIZE > MATERIAL_PAYLOAD_LIMIT
        let data = data_buffer.load<$MxMaterialDataName>(0);
#endif
        float3 opacity_weighted_emission;
        setup_material_instance_impl(data, si, lod_sampler, hints, opacity_weighted_emission);
        return opacity_weighted_emission;
#else
        return float3(0.0);
#endif
    }
    public override float3 eval_emission<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
#if MATERIALX_HAS_EMISSION
        return eval_emissive_radiance(si, lod_sampler, hints) * float.PI;
#else
        return float3(0.0);
#endif
    }
};
