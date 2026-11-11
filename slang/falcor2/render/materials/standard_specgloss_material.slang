// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct StandardSpecGlossMaterial : IMaterial
{
    typedef StandardMaterialInstance MaterialInstance;

    public MaterialHeader header;
    public TextureHandle diffuse_texture;
    public TextureHandle specular_glossiness_texture;
    public TextureHandle normal_texture;
    public TextureHandle emissive_texture;
    public TextureHandle transmission_texture;
    // TODO: These should be float16_t3 but CUDA backend has inconsistent reflection and dynamic dispatch layout rules!
    public float16_t4 diffuse_factor;
    public float16_t4 specular_factor;
    public float16_t4 transmission_factor;
    public float16_t glossiness_factor;
    public float16_t normal_texture_scale;
    public float16_t ior;
    public uint texture_coordinate_selectors;
    public float16_t diffuse_transmission_factor;
    public float16_t specular_transmission_factor;
    public float3 emissive_factor;
    public float3 volume_sigma_a;
    public float3 volume_sigma_s;
    public float volume_anisotropy;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};

        // Build the material shading frame, applying the normal map unless the caller disabled it.
        if (normal_texture.is_valid() && surface_material::is_normal_mapping_enabled(hints)) {
            const float2 texture_coord
                = surface_material::resolve_texcoord(si, stdmat::TCBits::normal::extract(texture_coordinate_selectors));
            float3 normal = surface_material::decode_normal_map(
                normal_texture.sample(lod_sampler, texture_coord, float4(0.0, 0.0, 1.0, 0.0)).xyz,
                stdmat::TCBits::normal_map_encoded::extract(texture_coordinate_selectors) != 0
            );
            normal = normalize(lerp(float3(0.0, 0.0, 1.0), normal, float(normal_texture_scale)));
            mi.shading_frame_ws = Frame(
                si.shading_frame_ws.to_global(normal),
                si.shading_frame_ws.tangent,
                si.shading_frame_ws.handedness_sign()
            );
        } else {
            mi.shading_frame_ws = si.shading_frame_ws;
        }
        const bool orient_backface = is_set(header.flags, MaterialFlags::two_sided);
        mi.shading_frame_ws = surface_material::orient_shading_frame(si, mi.shading_frame_ws, orient_backface);
        // TODO: Use is_set() once https://github.com/shader-slang/slang/issues/12926 is fixed.
        if ((hints & MaterialInstanceHints::adjust_shading_normal) != MaterialInstanceHints::none) {
            mi.shading_frame_ws = surface_material::adjust_shading_frame(si, mi.shading_frame_ws, orient_backface);
        }

        // Resolve textured glTF spec/gloss inputs into the shared BSDF parameterization.
        float3 diffuse;
        {
            const float2 texture_coord = surface_material::resolve_texcoord(
                si,
                stdmat::TCBits::diffuse::extract(texture_coordinate_selectors)
            );
            diffuse = diffuse_texture.sample(lod_sampler, texture_coord, float4(1.0)).rgb * diffuse_factor.xyz;
            if (stdmat::TCBits::use_vertex_color::extract(texture_coordinate_selectors) != 0)
                diffuse *= surface_material::resolve_vertex_color(si).rgb;
        }
        float4 specular_glossiness;
        {
            const float2 texture_coord = surface_material::resolve_texcoord(
                si,
                stdmat::TCBits::specular_glossiness::extract(texture_coordinate_selectors)
            );
            specular_glossiness = specular_glossiness_texture.sample(lod_sampler, texture_coord, float4(1.0))
                * float4(specular_factor.x, specular_factor.y, specular_factor.z, glossiness_factor);
        }
        float3 specular = saturate(specular_glossiness.rgb);
        float glossiness = saturate(specular_glossiness.a);

        mi.interior_ior = max(ior, 1e-3);

        StandardBSDFData bsdf_data = {};
        // glTF spec/gloss reserves diffuse energy for the resolved specular reflectance.
        bsdf_data.diffuse = diffuse * (1.0 - max(specular.x, max(specular.y, specular.z)));
        bsdf_data.specular = specular;
        bsdf_data.roughness = 1.0 - glossiness;
        bsdf_data.metallic = stdmat::get_metallic(diffuse, specular);
        bsdf_data.eta = si.front_facing ? (si.exterior_ior / mi.interior_ior) : (mi.interior_ior / si.exterior_ior);
        {
            const float2 texture_coord = surface_material::resolve_texcoord(
                si,
                stdmat::TCBits::transmission::extract(texture_coordinate_selectors)
            );
            bsdf_data.transmission
                = transmission_texture.sample(lod_sampler, texture_coord, float4(1.0)).rgb * transmission_factor.xyz;
        }
        bsdf_data.diffuse_transmission = saturate(diffuse_transmission_factor);
        bsdf_data.specular_transmission = saturate(specular_transmission_factor);
        bsdf_data.thin_walled = header.flags & MaterialFlags::thin_walled;
        mi.bsdf_data = bsdf_data;

        {
            const float2 texture_coord = surface_material::resolve_texcoord(
                si,
                stdmat::TCBits::emissive::extract(texture_coordinate_selectors)
            );
            mi.emissive_radiance
                = emissive_texture.sample(lod_sampler, texture_coord, float4(1.0)).rgb * emissive_factor;
        }
        mi.volume_sigma_a = volume_sigma_a;
        mi.volume_sigma_s = volume_sigma_s;
        mi.volume_anisotropy = volume_anisotropy;

        return mi;
    }

    public override bool is_emissive() { return any(emissive_factor > 0.0); }

    public override MaterialFlags flags() { return header.flags; }

    public override float eval_opacity<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        if (!is_set(header.opacity_flags, OpacityFlags::enabled))
            return 1.0;

        const uint texture_coord_index = stdmat::TCBits::diffuse::extract(texture_coordinate_selectors);
        const float2 texture_coord = surface_material::resolve_texcoord(si, texture_coord_index);
        float opacity = diffuse_texture.sample(lod_sampler, texture_coord, float4(1.0)).a * header.opacity_factor;
        if (stdmat::TCBits::use_vertex_color::extract(texture_coordinate_selectors) != 0)
            opacity *= surface_material::resolve_vertex_color(si).a;
        return opacity;
    }

    public override uint2 compute_emission_sampling_grid(float2 uv[3]) { return uint2(100, 100); }

    public override float3 eval_emission<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        const uint texture_coord_index = stdmat::TCBits::emissive::extract(texture_coordinate_selectors);
        const float2 texture_coord = surface_material::resolve_texcoord(si, texture_coord_index);
        return emissive_texture.sample(lod_sampler, texture_coord, float4(1.0)).rgb * emissive_factor * float.PI;
    }

    public override float3 eval_emissive_radiance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        const uint texture_coord_index = stdmat::TCBits::emissive::extract(texture_coordinate_selectors);
        const float2 texture_coord = surface_material::resolve_texcoord(si, texture_coord_index);
        return emissive_texture.sample(lod_sampler, texture_coord, float4(1.0)).rgb * emissive_factor;
    }
};
