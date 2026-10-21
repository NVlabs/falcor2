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
    public uint base_color_texture_texcoord;
    public float3 base_color_factor;
    public bool use_vertex_color;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};
        const float2 texture_coord = surface_material::resolve_texcoord(si, base_color_texture_texcoord);
        float4 vertex_color = float4(1.0);
        if (use_vertex_color)
            vertex_color = surface_material::resolve_vertex_color(si);
        mi.color = base_color_texture.sample(lod_sampler, texture_coord, float4(1.0)).rgb * base_color_factor
            * vertex_color.rgb;
        return mi;
    }

    public override MaterialFlags flags() { return header.flags; }

    public override float eval_opacity<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        if (!is_set(header.opacity_flags, OpacityFlags::enabled))
            return 1.0;

        const float2 texture_coord = surface_material::resolve_texcoord(si, base_color_texture_texcoord);
        float opacity = base_color_texture.sample(lod_sampler, texture_coord, float4(1.0)).a * header.opacity_factor;
        if (use_vertex_color)
            opacity *= surface_material::resolve_vertex_color(si).a;
        return opacity;
    }
};
