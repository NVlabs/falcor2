// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct GLTFSpecGlossMaterialInstance : IMaterialInstance
{
    public Frame shading_frame_ws;
    public GLTFSpecGlossBSDFData bsdf_data;
    public float3 emissive_radiance = {};

    private GLTFSpecGlossBSDF make_bsdf(const SurfaceInteraction si)
    {
        return GLTFSpecGlossBSDF(shading_frame_ws.to_local(si.wi_ws), bsdf_data);
    }

    public BSDFEval eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both, S : ISampleGenerator>(
        const SurfaceInteraction si,
        const float3 wo_ws,
        inout S sg
    )
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);

        // This material only reflects. Reject directions below the shading horizon
        // or on the wrong side of the geometric surface.
        if (wo_ls.z < BSDF_MIN_COS_THETA
            || !surface_material::is_valid_hemisphere_reflection_or_transmission(
                si,
                shading_frame_ws,
                wi_ls,
                wo_ls,
                wo_ws
            ))
            return BSDFEval();

        // The BSDF already includes the outgoing cosine in its evaluated value.
        GLTFSpecGlossBSDF bsdf = GLTFSpecGlossBSDF(wi_ls, bsdf_data);
        return bsdf.eval_with_pdf(wi_ls, wo_ls, sg, BSDFContext());
    }

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out MaterialSample sample)
    {
        sample = {};
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        GLTFSpecGlossBSDF bsdf = GLTFSpecGlossBSDF(wi_ls, bsdf_data);
        BSDFSample bsdf_sample;
        if (!bsdf.sample(wi_ls, bsdf_sample, sg, BSDFContext()))
            return false;

        // Preserve the BSDF's component-sampling weight and mixture PDF.
        // An orthonormal frame change preserves solid angle, so neither needs rescaling.
        sample.pdf = bsdf_sample.pdf;
        sample.weight = bsdf_sample.weight;
        sample.flags = bsdf_sample.flags;
        sample.wo_ws = shading_frame_ws.to_global(bsdf_sample.wo);

        // Invalid samples contribute zero; do not renormalize the remaining PDF.
        return surface_material::is_valid_hemisphere_reflection_or_transmission(
            si,
            shading_frame_ws,
            wi_ls,
            bsdf_sample.wo,
            sample.wo_ws
        );
    }

    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = {};

        // glTF defines alpha = (1 - glossiness)^2. Guides use perceptual roughness,
        // with zero marking the same delta approximation used by the BSDF.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/README.md#specular---glossiness
        float roughness = 1.0 - saturate(bsdf_data.glossiness);
        float alpha = roughness * roughness;
        float3 specular = saturate(bsdf_data.specular);
        result.material_color = saturate(bsdf_data.diffuse);
        result.emission = emissive_radiance;

        // SpecGloss has no authored metallic parameter. Report the actual lobe colors.
        result.metallic = 0.0;
        result.roughness = alpha < GLTF_MIN_GGX_ALPHA ? 0.0 : roughness;
        result.guide_normal = shading_frame_ws.normal;

        // Integrating Lambert's cosine-weighted BRDF cancels 1/pi. The remaining
        // color is diffuse * (1 - max(F0)).
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/README.md#specular---glossiness
        result.diffuse_reflection_albedo = bsdf_data.diffuse * (1.0 - max(specular.x, max(specular.y, specular.z)));

        // F0 is used as a view-independent specular albedo approximation here;
        // it does not include the increased Fresnel reflection at grazing angles.
        result.specular_reflection_albedo = specular;
        result.specular_reflectance = specular;
        return result;
    }

    public override Optional<ExtraBsdfProperties>
    collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        ExtraBsdfProperties result = {};
        MaterialProperties properties = collect_properties(si);

        // Two conceptual lobes share the normal-mapped frame. Report its axes
        // relative to the surface interaction's frame, as the extra-property API expects.
        result.bsdf_count = 2;
        for (uint i = 0; i < 2; ++i) {
            result.bsdf_N[i] = si.shading_frame_ws.to_local(shading_frame_ws.normal);
            result.bsdf_T[i] = si.shading_frame_ws.to_local(shading_frame_ws.tangent);
            result.bsdf_B[i] = si.shading_frame_ws.to_local(shading_frame_ws.bitangent);

            // The reported albedos already include the lobe scaling.
            result.bsdf_weight[i] = float3(1.0);
        }

        // Lambert has no roughness parameter; one is its guide convention.
        result.bsdf_albedo[0] = properties.diffuse_reflection_albedo;
        result.bsdf_roughness[0] = float2(1.0);

        // The isotropic specular guide uses equal perceptual roughness on both axes,
        // following StandardMaterial's extra-property convention.
        result.bsdf_albedo[1] = properties.specular_reflection_albedo;
        result.bsdf_roughness[1] = float2(properties.roughness);
        return result;
    }

    public BSDFFlags get_lobe_types(const SurfaceInteraction si) { return make_bsdf(si).get_lobe_types(); }

    public override bool is_emissive() { return any(emissive_radiance > 0.0); }

    public override float3 eval_emissive_radiance() { return emissive_radiance; }
};
