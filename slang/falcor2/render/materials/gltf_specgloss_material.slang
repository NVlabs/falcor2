// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

// Keep the four core texture infos physically flat. Logical accessors share
// the glTF sampling code without the measured cost of nested core fields.
public struct GLTFSpecGlossMaterial<
    let Features : uint,
    let Textures : uint,
    let Transforms : uint,
    let NonzeroTexcoords : uint> : IMaterial
{
    typedef GLTFSpecGlossMaterialInstance<Features> MaterialInstance;

    public static const bool HAS_EMISSION = (Features & gltfmat::FeatureBits::EMISSION) != 0u;
    public static const bool HAS_NORMAL = (Features & gltfmat::FeatureBits::NORMAL) != 0u;
    public static const bool HAS_VERTEX_COLOR = (Features & gltfmat::FeatureBits::VERTEX_COLOR) != 0u;

    public MaterialHeader header;
    public TextureHandle diffuse_handle;
    public uint diffuse_packed;
    public TextureHandle specular_glossiness_handle;
    public uint specular_glossiness_packed;
    public TextureHandle normal_handle;
    public uint normal_packed;
    public TextureHandle emissive_handle;
    public uint emissive_packed;
    // CUDA dynamic dispatch and reflection require padded half vectors.
    public float16_t4 diffuse_factor;
    public float16_t4 specular_factor;
    public float4 emissive_factor;
    // Glossiness, normal scale, reserved, reserved.
    public float16_t4 core_factors;
    public uint feature_mask;
    public uint transform_mask;
    public BufferPtr extra_data;

    private GLTFTextureInfo diffuse_info()
    {
        GLTFTextureInfo info = {};
        info.handle = diffuse_handle;
        info.packed = diffuse_packed;
        return info;
    }

    private GLTFTextureInfo specular_glossiness_info()
    {
        GLTFTextureInfo info = {};
        info.handle = specular_glossiness_handle;
        info.packed = specular_glossiness_packed;
        return info;
    }

    private GLTFTextureInfo normal_info()
    {
        GLTFTextureInfo info = {};
        info.handle = normal_handle;
        info.packed = normal_packed;
        return info;
    }

    private GLTFTextureInfo emissive_info()
    {
        GLTFTextureInfo info = {};
        info.handle = emissive_handle;
        info.packed = emissive_packed;
        return info;
    }

    private float4 sample_texture<let SlotBit : uint, LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        GLTFTextureInfo info,
        float4 fallback
    )
    {
        return gltf_material::sample_texture_sparse<Textures, Transforms, NonzeroTexcoords, SlotBit>(
            si,
            lod_sampler,
            info,
            feature_mask,
            transform_mask,
            extra_data,
            fallback
        );
    }

    // Resolve authored diffuse, specular F0, and glossiness at this hit without converting to metallic-roughness.
    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/README.md#specular---glossiness
    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};
        mi.feature_mask = feature_mask;

        GLTFTextureInfo normal_texture = {};
        if (HAS_NORMAL) {
            if ((feature_mask & gltfmat::FeatureBits::NORMAL) != 0u)
                normal_texture = normal_info();
        }
        // The shared helper still orients backfaces and adjusts the shading
        // normal when the normal texture is absent.
        mi.shading_frame_ws = gltf_material::
            shading_frame_sparse<Textures, Transforms, NonzeroTexcoords, gltfmat::SpecGlossTextureBits::NORMAL>(
                si,
                lod_sampler,
                hints,
                header.flags,
                normal_texture,
                float(core_factors.y),
                feature_mask,
                transform_mask,
                extra_data
            );

        // Sample linear RGB and multiply by the authored diffuse factor. White is the missing-texture identity.
        float3 diffuse
            = sample_texture<gltfmat::SpecGlossTextureBits::DIFFUSE>(si, lod_sampler, diffuse_info(), float4(1.0)).rgb
            * diffuse_factor.xyz;

        // Vertex RGB modulates diffuse only; vertex alpha contributes to coverage in eval_opacity().
        if (HAS_VERTEX_COLOR) {
            if ((feature_mask & gltfmat::FeatureBits::VERTEX_COLOR) != 0u)
                diffuse *= surface_material::resolve_vertex_color(si).rgb;
        }

        // This texture combines sRGB specular color with linear glossiness in alpha.
        // The color texture view decodes RGB to linear while leaving alpha unchanged.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/README.md#specularglossinesstexture
        const float4 specular_glossiness = sample_texture<gltfmat::SpecGlossTextureBits::SPECULAR_GLOSSINESS>(
            si,
            lod_sampler,
            specular_glossiness_info(),
            float4(1.0)
        );

        // Defer the diffuse reduction, diffuse * (1 - max(F0.r, F0.g, F0.b)), to GLTFSpecGlossBSDF.
        mi.bsdf_data.diffuse = diffuse;

        // Specular RGB is already F0: do not multiply by 0.04 or derive it from an IOR.
        mi.bsdf_data.specular = saturate(specular_glossiness.rgb * specular_factor.xyz);

        // Keep authored glossiness here. The BSDF computes roughness = 1 - glossiness, then alpha = roughness^2.
        mi.bsdf_data.glossiness = saturate(specular_glossiness.a * float(core_factors.x));

        // Emission is independent of diffuse and specular scattering.
        if (HAS_EMISSION) {
            if ((feature_mask & gltfmat::FeatureBits::EMISSION) != 0u)
                mi.emission.set(eval_emissive_radiance(si, lod_sampler, hints));
        }
        return mi;
    }

    public override MaterialFlags flags() { return header.flags; }
    public override bool is_emissive()
    {
        if (HAS_EMISSION) {
            if ((feature_mask & gltfmat::FeatureBits::EMISSION) != 0u)
                return any(emissive_factor.xyz > 0.0);
        }
        return false;
    }

    // Coverage uses diffuse alpha, the header opacity factor, and optional vertex alpha.
    // It is handled separately from the reflective BSDF, not as a transmission lobe.
    // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#alpha-coverage
    public override float eval_opacity<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        if (!is_set(header.opacity_flags, OpacityFlags::enabled))
            return 1.0;
        float result = header.opacity_factor;
        result
            *= sample_texture<gltfmat::SpecGlossTextureBits::DIFFUSE>(si, lod_sampler, diffuse_info(), float4(1.0)).a;
        if (HAS_VERTEX_COLOR) {
            if ((feature_mask & gltfmat::FeatureBits::VERTEX_COLOR) != 0u)
                result *= surface_material::resolve_vertex_color(si).a;
        }
        return result;
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
        if (HAS_EMISSION) {
            if ((feature_mask & gltfmat::FeatureBits::EMISSION) != 0u) {
                // Resolve UV only for a valid texture and multiply its linear
                // RGB by the authored emission factor.
                return sample_texture<gltfmat::SpecGlossTextureBits::EMISSIVE>(
                           si,
                           lod_sampler,
                           emissive_info(),
                           float4(1.0)
                       )
                           .rgb
                    * emissive_factor.xyz;
            }
        }
        return float3(0.0);
    }
};
