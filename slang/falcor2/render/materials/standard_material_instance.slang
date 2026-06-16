// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct StandardMaterialInstance : IMaterialInstance
{
    public Frame shading_frame_ws;
    public StandardBSDFData bsdf_data;
    public float3 emissive_radiance;

    private StandardBSDF make_bsdf(const SurfaceInteraction si)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        return StandardBSDF(wi_ls, bsdf_data);
    }

    public float3 eval<S : ISampleGenerator>(const SurfaceInteraction si, const float3 wo_ws, inout S sg)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);
        if (!stdmat::is_valid_hemisphere_reflection_or_transmission(si, shading_frame_ws, wi_ls, wo_ls, wo_ws))
            return float3(0.0);

        StandardBSDF bsdf = StandardBSDF(wi_ls, bsdf_data);
        return bsdf.eval(wi_ls, wo_ls, sg, BSDFContext());
    }

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out BSDFSample result)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls;
        StandardBSDF bsdf = StandardBSDF(wi_ls, bsdf_data);
        result = {};
        bool valid = bsdf.sample(wi_ls, wo_ls, result.pdf, result.weight, result.lobe_types, sg, BSDFContext());
        if (!valid)
            return false;

        result.wo_ws = shading_frame_ws.to_global(wo_ls);
        if (!stdmat::is_valid_hemisphere_reflection_or_transmission(si, shading_frame_ws, wi_ls, wo_ls, result.wo_ws)) {
            result = {};
            return false;
        }

        return true;
    }

    public float eval_pdf(const SurfaceInteraction si, const float3 wo_ws)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);
        if (!stdmat::is_valid_hemisphere_reflection_or_transmission(si, shading_frame_ws, wi_ls, wo_ls, wo_ws))
            return 0.0;

        StandardBSDF bsdf = StandardBSDF(wi_ls, bsdf_data);
        return bsdf.eval_pdf(wi_ls, wo_ls, BSDFContext());
    }

    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = {};

        // Convert perceptual roughness to the guide value used by downstream denoisers.
        float alpha = bsdf_data.roughness * bsdf_data.roughness;
        float roughness = alpha < STANDARD_MIN_GGX_ALPHA ? 0.0 : bsdf_data.roughness;

        // Split the material into the guide albedo buckets expected by the renderer.
        result.emission = emissive_radiance;
        result.roughness = roughness;
        result.guide_normal = shading_frame_ws.normal;
        result.diffuse_reflection_albedo
            = (1.0 - bsdf_data.diffuse_transmission) * (1.0 - bsdf_data.specular_transmission) * bsdf_data.diffuse;
        result.diffuse_transmission_albedo
            = bsdf_data.diffuse_transmission * (1.0 - bsdf_data.specular_transmission) * bsdf_data.transmission;
        result.specular_reflection_albedo = (1.0 - bsdf_data.specular_transmission) * bsdf_data.specular;
        result.specular_transmission_albedo = bsdf_data.specular_transmission * bsdf_data.specular;
        result.specular_reflectance = bsdf_data.specular;
        if (bsdf_data.diffuse_transmission > 0.0 || bsdf_data.specular_transmission > 0.0)
            result.flags = MaterialProperties::Flags::is_transmissive;

        return result;
    }

    public override float3 eval_material_color(const SurfaceInteraction si)
    {
        // Give us some form of 'color' even for materials that are purely specular,
        // to help guide denoisers and other post-processes.
        float metallic = saturate(bsdf_data.metallic);
        float3 dielectric_color = metallic < 0.999 ? bsdf_data.diffuse / max(1.0 - metallic, 1e-3) : bsdf_data.specular;
        return saturate(lerp(dielectric_color, bsdf_data.specular, metallic));
    }

    public override Optional<ExtraBsdfProperties>
    collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        ExtraBsdfProperties result = {};

        // Use a delta marker for perfectly smooth specular lobes.
        float alpha = bsdf_data.roughness * bsdf_data.roughness;
        float roughness = alpha < STANDARD_MIN_GGX_ALPHA ? 0.0 : bsdf_data.roughness;

        // Express the material-local frame relative to the geometric shading frame.
        float3 local_n = si.shading_frame_ws.to_local(shading_frame_ws.normal);
        float3 local_t = si.shading_frame_ws.to_local(shading_frame_ws.tangent);
        float3 local_b = si.shading_frame_ws.to_local(shading_frame_ws.bitangent);

        // Expose the same four conceptual lobes that StandardBSDF samples.
        result.bsdf_count = 4;

        result.bsdf_N[0] = local_n;
        result.bsdf_T[0] = local_t;
        result.bsdf_B[0] = local_b;
        result.bsdf_albedo[0] = bsdf_data.diffuse;
        result.bsdf_weight[0]
            = float3((1.0 - bsdf_data.specular_transmission) * (1.0 - bsdf_data.diffuse_transmission));
        result.bsdf_roughness[0] = float2(1.0);

        result.bsdf_N[1] = local_n;
        result.bsdf_T[1] = local_t;
        result.bsdf_B[1] = local_b;
        result.bsdf_albedo[1] = bsdf_data.transmission;
        result.bsdf_weight[1] = float3((1.0 - bsdf_data.specular_transmission) * bsdf_data.diffuse_transmission);
        result.bsdf_roughness[1] = float2(1.0);

        result.bsdf_N[2] = local_n;
        result.bsdf_T[2] = local_t;
        result.bsdf_B[2] = local_b;
        result.bsdf_albedo[2] = bsdf_data.specular;
        result.bsdf_weight[2] = float3(1.0 - bsdf_data.specular_transmission);
        result.bsdf_roughness[2] = float2(roughness);

        result.bsdf_N[3] = local_n;
        result.bsdf_T[3] = local_t;
        result.bsdf_B[3] = local_b;
        result.bsdf_albedo[3] = bsdf_data.specular;
        result.bsdf_weight[3] = float3(bsdf_data.specular_transmission);
        result.bsdf_roughness[3] = float2(roughness);

        return result;
    }

    public LobeTypes get_lobe_types(const SurfaceInteraction si)
    {
        StandardBSDF bsdf = make_bsdf(si);
        return bsdf.get_lobe_types();
    }

    public override bool is_emissive() { return any(emissive_radiance > 0.0); }

    public override float3 eval_emissive_radiance() { return emissive_radiance; }
};
