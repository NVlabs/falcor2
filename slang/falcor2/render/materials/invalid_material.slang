// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct InvalidMaterialInstance : IMaterialInstance
{
    public Frame shading_frame_ws;
    public LambertDiffuseBRDF brdf;

    public float3 eval<S : ISampleGenerator>(const SurfaceInteraction si, const float3 wo_ws, inout S sg)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);
        return brdf.eval(wi_ls, wo_ls, sg, BSDFContext());
    }

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out BSDFSample result)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls;
        result = {};
        bool valid = brdf.sample(wi_ls, wo_ls, result.pdf, result.weight, result.lobe_types, sg, BSDFContext());
        result.wo_ws = shading_frame_ws.to_global(wo_ls);
        return valid;
    }

    public float eval_pdf(const SurfaceInteraction si, const float3 wo_ws)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);
        return brdf.eval_pdf(wi_ls, wo_ls, BSDFContext());
    }

    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        let albedo = brdf.eval_albedo(wi_ls);

        MaterialProperties result = {};
        result.guide_normal = shading_frame_ws.normal;
        result.diffuse_reflection_albedo = albedo.reflection;
        return result;
    }

    public override Optional<ExtraBsdfProperties>
    collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        ExtraBsdfProperties result;
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        result.bsdf_count = 1;
        result.bsdf_N[0] = float3(0, 0, 1);
        result.bsdf_T[0] = float3(1, 0, 0);
        result.bsdf_B[0] = float3(0, 1, 0);
        AlbedoContributions a = brdf.eval_albedo(wi_ls);
        result.bsdf_albedo[0] = a.reflection;
        result.bsdf_weight[0] = float3(1, 1, 1);
        return result;
    }

    public LobeTypes get_lobe_types(const SurfaceInteraction si) { return LobeTypes::diffuse_reflection; }
};

public struct InvalidMaterial : IMaterial
{
    typedef InvalidMaterialInstance MaterialInstance;

    public MaterialHeader header;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};
        mi.shading_frame_ws = si.shading_frame_ws;
        if (hints & MaterialInstanceHints::adjust_shading_normal) {
            adjust_shading_normal(mi.shading_frame_ws, si.normal_ws, si.wi_ws);
        }
        mi.brdf = LambertDiffuseBRDF(float3(1.0, 0.0, 1.0));
        return mi;
    }
};
