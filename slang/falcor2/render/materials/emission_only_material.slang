// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct EmissionOnlyMaterialInstance : IMaterialInstance
{
    public float3 emissive_radiance;


    public BSDFEval eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both, S : ISampleGenerator>(
        const SurfaceInteraction si,
        const float3 wo_ws,
        inout S sg
    )
    {
        return BSDFEval();
    }

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out MaterialSample sample)
    {
        sample = {};
        return false;
    }


    public override bool is_emissive() { return any(emissive_radiance > 0.0); }

    public override float3 eval_emissive_radiance() { return emissive_radiance; }

    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = {};
        result.emission = emissive_radiance;
        return result;
    }

    public BSDFFlags get_lobe_types(const SurfaceInteraction si) { return BSDFFlags::none; }
};

public struct EmissionOnlyMaterial : IMaterial
{
    typedef EmissionOnlyMaterialInstance MaterialInstance;

    public MaterialHeader header;
    public TextureHandle emission_color_texture;
    public float3 emission_color;
    public float emission_luminance;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};
        mi.emissive_radiance
            = emission_luminance * emission_color * emission_color_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb;
        return mi;
    }

    public override bool is_emissive() { return emission_luminance > 0.0 && any(emission_color > 0.0); }

    public override MaterialFlags flags() { return header.flags; }

    public override uint2 compute_emission_sampling_grid(float2 uv[3])
    {
        return emission_color_texture.is_valid() ? uint2(100, 100) : uint2(1, 1);
    }

    public override float3 eval_emission<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        return emission_luminance * emission_color * emission_color_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb
            * float.PI;
    }

    public override float3 eval_emissive_radiance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        return emission_luminance * emission_color * emission_color_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb;
    }
};
