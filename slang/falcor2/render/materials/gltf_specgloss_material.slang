// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct GLTFSpecGlossMaterial : IMaterial
{
    typedef GLTFSpecGlossMaterialInstance MaterialInstance;

    public MaterialHeader header;
    public TextureHandle diffuse_texture;
    public TextureHandle normal_texture;
    public TextureHandle specular_glossiness_texture;
    public TextureHandle emissive_texture;
    // Keep half vectors padded consistently with the other native materials on CUDA.
    public float16_t4 diffuse_factor;
    public float16_t4 specular_factor;
    public float16_t glossiness_factor;
    public float16_t normal_texture_scale;
    public uint texture_coordinate_selectors;
    public float3 emissive_factor;
    public uint transform_mask;
    public BufferOffsetPair texture_transform_data;

    // Resolve authored diffuse, specular F0, and glossiness at this hit without converting to metallic-roughness.
    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/README.md#specular---glossiness
    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};

        // Use the shared glTF normal decoding, XY scale, sidedness, and optional shading-normal adjustment.
        // Transform records start at byte zero; binding 2 is the normal texture, not the selected UV set.
        mi.shading_frame_ws = gltf_material::shading_frame(
            si,
            lod_sampler,
            hints,
            header.flags,
            normal_texture,
            float(normal_texture_scale),
            gltfmat::SpecGlossTCBits::normal::extract(texture_coordinate_selectors),
            transform_mask,
            texture_transform_data,
            0,
            2,
            gltfmat::SpecGlossTCBits::normal_map_encoded::extract(texture_coordinate_selectors) != 0
        );

        // Sample linear RGB and multiply by the authored diffuse factor. White is the missing-texture identity.
        const float2 diffuse_uv = gltf_material::resolve_texcoord(
            si,
            gltfmat::SpecGlossTCBits::diffuse::extract(texture_coordinate_selectors),
            0,
            transform_mask,
            texture_transform_data
        );
        float3 diffuse = diffuse_texture.sample(lod_sampler, diffuse_uv, float4(1.0)).rgb * diffuse_factor.xyz;

        // Vertex RGB modulates diffuse only; vertex alpha contributes to coverage in eval_opacity().
        if (gltfmat::SpecGlossTCBits::use_vertex_color::extract(texture_coordinate_selectors) != 0)
            diffuse *= surface_material::resolve_vertex_color(si).rgb;

        // This texture combines sRGB specular color with linear glossiness in alpha.
        // The color texture view decodes RGB to linear while leaving alpha unchanged.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/README.md#specularglossinesstexture
        const float2 specular_uv = gltf_material::resolve_texcoord(
            si,
            gltfmat::SpecGlossTCBits::specular_glossiness::extract(texture_coordinate_selectors),
            1,
            transform_mask,
            texture_transform_data
        );
        const float4 specular_glossiness = specular_glossiness_texture.sample(lod_sampler, specular_uv, float4(1.0));

        // Defer the diffuse reduction, diffuse * (1 - max(F0.r, F0.g, F0.b)), to GLTFSpecGlossBSDF.
        mi.bsdf_data.diffuse = diffuse;

        // Specular RGB is already F0: do not multiply by 0.04 or derive it from an IOR.
        mi.bsdf_data.specular = saturate(specular_glossiness.rgb * specular_factor.xyz);

        // Keep authored glossiness here. The BSDF computes roughness = 1 - glossiness, then alpha = roughness^2.
        mi.bsdf_data.glossiness = saturate(specular_glossiness.a * glossiness_factor);

        // Emission is independent of diffuse and specular scattering.
        mi.emissive_radiance = eval_emissive_radiance(si, lod_sampler, hints);
        return mi;
    }

    public override MaterialFlags flags() { return header.flags; }
    public override bool is_emissive() { return any(emissive_factor > 0.0); }

    // Coverage uses diffuse alpha, the header opacity factor, and optional vertex alpha.
    // It is handled separately from the reflective BSDF, not as a transmission lobe.
    // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#alpha-coverage
    public override float eval_opacity<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        return gltf_material::opacity(
            si,
            lod_sampler,
            header,
            diffuse_texture,
            gltfmat::SpecGlossTCBits::diffuse::extract(texture_coordinate_selectors),
            gltfmat::SpecGlossTCBits::use_vertex_color::extract(texture_coordinate_selectors) != 0,
            transform_mask,
            texture_transform_data
        );
    }

    // Fixed resolution for sampling the emitter's texture; this is a sampling policy, not a material parameter.
    public override uint2 compute_emission_sampling_grid(float2 uv[3]) { return uint2(100, 100); }

    public override float3 eval_emission<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        // Direction-independent radiance integrates to exitance M = pi * L over the emitting hemisphere.
        return eval_emissive_radiance(si, lod_sampler, hints) * float.PI;
    }

    public override float3 eval_emissive_radiance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        // Resolve the emissive UV transform, then multiply linear texture RGB by the emission factor.
        return gltf_material::emission(
            si,
            lod_sampler,
            emissive_texture,
            emissive_factor,
            gltfmat::SpecGlossTCBits::emissive::extract(texture_coordinate_selectors),
            transform_mask,
            texture_transform_data
        );
    }
};
