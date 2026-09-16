// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct UnlitMaterialInstance : IMaterialInstance
{
    public float3 color;


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


    // Special case: The path tracer treats unlit color as terminal emitted radiance for beauty and guide output,
    // while is_emissive() remains false so the material does not participate in light sampling.
    public override float3 eval_emissive_radiance() { return color; }

    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = {};
        result.material_color = color;
        result.emission = color;
        return result;
    }

    public BSDFFlags get_lobe_types(const SurfaceInteraction si) { return BSDFFlags::none; }
};

public struct UnlitMaterial : IMaterial
{
    typedef UnlitMaterialInstance MaterialInstance;

    public MaterialHeader header;
    public TextureHandle base_color_texture;
    public float3 base_color_factor;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};
        mi.color = base_color_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb * base_color_factor;
        return mi;
    }

    public override MaterialFlags flags() { return header.flags; }
};
