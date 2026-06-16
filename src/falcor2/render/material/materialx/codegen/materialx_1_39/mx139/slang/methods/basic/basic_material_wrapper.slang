// SPDX-License-Identifier: Apache-2.0

public struct $MxMaterialInstanceName : IMaterialInstance
{
    typealias mtlx_ns = $MxAliasesName;
    public Frame shading_frame_ws;
    public mtlx_ns::RootBSDF bsdf;
    public float3 emissive_radiance = {};
    public bool thin_walled = false;
    public float3 eval<S : ISampleGenerator>(const SurfaceInteraction si, const float3 wo_ws, inout S sg)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);
        return bsdf.eval(wi_ls, wo_ls, sg, BSDFContext());
    }
    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out BSDFSample result)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls;
        result = {};
        bool valid = bsdf.sample(wi_ls, wo_ls, result.pdf, result.weight, result.lobe_types, sg, BSDFContext());
        result.wo_ws = shading_frame_ws.to_global(wo_ls);
        return valid;
    }
    public float eval_pdf(const SurfaceInteraction si, const float3 wo_ws)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);
        return bsdf.eval_pdf(wi_ls, wo_ls, BSDFContext());
    }
    public override bool is_emissive() { return any(emissive_radiance > 0.0); }
    public override float3 eval_emissive_radiance() { return emissive_radiance; }
    public override bool is_thin_walled() { return thin_walled; }
    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        AlbedoContributions albedo = bsdf.eval_albedo(wi_ls);
        RoughnessInformation roughness = bsdf.eval_roughness(wi_ls);
        MaterialProperties result = {};
        result.roughness = roughness.roughness.x;
        result.guide_normal = shading_frame_ws.normal;
        result.specular_reflection_albedo = albedo.reflection;
        result.specular_transmission_albedo = albedo.transmission;
        result.specular_reflectance = bsdf.ior_as_reflectance;
        result.emission = emissive_radiance;
        if (thin_walled)
            result.flags = result.flags | MaterialProperties::Flags::is_thin_walled;
        $MxTransmissiveFlagLine$MxCurveScatteringFlagLine return result;
    }
    $MxCollectExtraBsdfPropertiesMethod public LobeTypes get_lobe_types(const SurfaceInteraction si)
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
    static MaterialInstance setup_material_instance<TLodSampler : ILodSampler>(
        $MxMaterialDataName data,
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints
    )
    {
        typealias Graph = $MxGraphName<TLodSampler>;
        MaterialInstance result = {};
        result.shading_frame_ws = si.shading_frame_ws;
        if (hints & MaterialInstanceHints::adjust_shading_normal) {
            adjust_shading_normal(result.shading_frame_ws, si.normal_ws, si.wi_ws);
        }
        SurfaceInteraction graph_si = si;
        graph_si.shading_frame_ws = result.shading_frame_ws;
        let mtlx_graph = Graph(data, graph_si, lod_sampler, uint(hints));
        $MxStackDecl;
        $MxSyntheticOpacityLines$MxRootValueLines result.bsdf = $MxRootValueExpr;
        result.emissive_radiance = $MxGraphEmission;
        result.thin_walled = $MxGraphThinWalled;
        return result;
    }
    MaterialInstance setup_material_instance<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints
    )
    {
#if MATERIALX_PAYLOAD_SIZE > MATERIAL_PAYLOAD_LIMIT
        let data = data_buffer.load_aligned<$MxMaterialDataName>(0, 16);
#endif
        return setup_material_instance(data, si, lod_sampler, hints);
    }
    public override bool is_emissive() { return $MxGraphCanEmit; }
    public override float3 eval_emissive_radiance<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        return setup_material_instance(si, lod_sampler, hints).eval_emissive_radiance();
    }
    public override float3 eval_emission<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        return eval_emissive_radiance(si, lod_sampler, hints) * float.PI;
    }
};
