// SPDX-License-Identifier: Apache-2.0

module checker_material;

__exported import falcor2.render;
import falcor2.utils;

public struct CheckerMaterialInstance : IMaterialInstance
{
    public Frame shading_frame_ws;
    public float3 albedo;
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
        MaterialProperties result = {};
        result.guide_normal = shading_frame_ws.normal;
        result.diffuse_reflection_albedo = albedo;
        return result;
    }

    public LobeTypes get_lobe_types(const SurfaceInteraction si) { return LobeTypes::diffuse_reflection; }
};

public struct CheckerMaterial : IMaterial
{
    typedef CheckerMaterialInstance MaterialInstance;

    public MaterialHeader header;
    public float3 color_a;
    public float3 color_b;
    public float scale;

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

        float2 uv = si.uv * scale;
        bool checker = ((int(floor(uv.x)) + int(floor(uv.y))) & 1) != 0;
        float3 albedo = checker ? color_b : color_a;

        mi.albedo = albedo;
        mi.brdf = LambertDiffuseBRDF(albedo);
        return mi;
    }
};
