// SPDX-License-Identifier: Apache-2.0

module example_python_material;
import falcor2.render;
import falcor2.utils;

public struct PythonMaterialSampleInstance : IMaterialInstance
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
        result.lobe_types = LobeTypes::diffuse_reflection;
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
        result.bsdf_roughness[0] = float2(0.5, 0.5);
        return result;
    }

    public LobeTypes get_lobe_types(const SurfaceInteraction si) { return LobeTypes::diffuse_reflection; }
};

public struct PythonMaterialSample : IMaterial
{
    public typealias MaterialInstance = PythonMaterialSampleInstance;
    public MaterialHeader header;
    public TextureHandle textures[2];
    public int texture_index;

    float3 get_albedo<TLodSampler : ILodSampler>(SurfaceInteraction si, TLodSampler lod_sampler, int index)
    {
        var albedo = float3(1, 0, 1);
        if (index >= 0 && index < 2)
            albedo = textures[index].sample(lod_sampler, si.uv).rgb;
        return albedo;
    }

    public MaterialInstance setup_material_instance<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};
        mi.shading_frame_ws = si.shading_frame_ws;
        if (hints & MaterialInstanceHints::adjust_shading_normal) {
            adjust_shading_normal(mi.shading_frame_ws, si.normal_ws, si.wi_ws);
        }

        mi.brdf = LambertDiffuseBRDF(get_albedo(si, lod_sampler, texture_index));
        return mi;
    }
};
