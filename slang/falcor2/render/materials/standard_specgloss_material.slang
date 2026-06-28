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
    public float16_t diffuse_transmission_factor;
    public float16_t specular_transmission_factor;
    public float3 emissive_factor;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};

        // Build the material shading frame, applying the normal map unless the caller disabled it.
        if (normal_texture.is_valid() && stdmat::is_normal_mapping_enabled(hints)) {
            float3 normal
                = normalize(normal_texture.sample(lod_sampler, si.uv, float4(0.0, 0.0, 1.0, 0.0)).xyz * 2.0 - 1.0);
            normal = normalize(lerp(float3(0.0, 0.0, 1.0), normal, float(normal_texture_scale)));
            mi.shading_frame_ws = Frame(
                si.shading_frame_ws.to_global(normal),
                si.shading_frame_ws.tangent,
                si.shading_frame_ws.handedness_sign()
            );
        } else {
            mi.shading_frame_ws = si.shading_frame_ws;
        }
        mi.shading_frame_ws
            = stdmat::orient_shading_frame(si, mi.shading_frame_ws, header.flags & MaterialFlags::double_sided);
        if (hints & MaterialInstanceHints::adjust_shading_normal) {
            adjust_shading_normal(mi.shading_frame_ws, si.front_normal_ws(), si.wi_ws);
        }

        // Resolve textured glTF spec/gloss inputs into the shared BSDF parameterization.
        float3 diffuse = diffuse_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb * diffuse_factor.xyz;
        float4 specular_glossiness = specular_glossiness_texture.sample(lod_sampler, si.uv, float4(1.0))
            * float4(specular_factor.x, specular_factor.y, specular_factor.z, glossiness_factor);
        float3 specular = saturate(specular_glossiness.rgb);
        float glossiness = saturate(specular_glossiness.a);

        float safe_ior = max(ior, 1e-3);

        StandardBSDFData bsdf_data = {};
        bsdf_data.diffuse = diffuse;
        bsdf_data.specular = specular;
        bsdf_data.roughness = 1.0 - glossiness;
        bsdf_data.metallic = stdmat::get_metallic(diffuse, specular);
        bsdf_data.eta = si.front_facing ? (si.ior / safe_ior) : (safe_ior / si.ior);
        bsdf_data.transmission
            = transmission_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb * transmission_factor.xyz;
        bsdf_data.diffuse_transmission = saturate(diffuse_transmission_factor);
        bsdf_data.specular_transmission = saturate(specular_transmission_factor);
        bsdf_data.thin_walled = header.flags & MaterialFlags::thin_walled;
        mi.bsdf_data = bsdf_data;

        mi.emissive_radiance = emissive_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb * emissive_factor;

        return mi;
    }

    public override bool is_emissive() { return any(emissive_factor > 0.0); }

    public override uint2 compute_emission_sampling_grid(float2 uv[3]) { return uint2(100, 100); }

    public override float3 eval_emission<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        return emissive_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb * emissive_factor * float.PI;
    }

    public override float3 eval_emissive_radiance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        return emissive_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb * emissive_factor;
    }
};
