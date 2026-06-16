// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct StandardMaterial : IMaterial
{
    typedef StandardMaterialInstance MaterialInstance;

    public MaterialHeader header;
    public TextureHandle base_color_texture;
    public TextureHandle normal_texture;
    public TextureHandle metallic_roughness_texture;
    public TextureHandle emissive_texture;
    public TextureHandle transmission_texture;
    public float3 base_color_factor;
    public float metallic_factor;
    public float roughness_factor;
    public float normal_texture_scale;
    public float3 emissive_factor;
    public float ior;
    public float3 transmission_factor;
    public float diffuse_transmission_factor;
    public float specular_transmission_factor;
    public bool thin_surface;
    public bool double_sided;
    /// Channel index for metallic value in metallic_roughness_texture.
    public uint metallic_texture_channel;
    /// Channel index for roughness value in metallic_roughness_texture.
    public uint roughness_texture_channel;

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
            normal = normalize(lerp(float3(0.0, 0.0, 1.0), normal, normal_texture_scale));
            mi.shading_frame_ws = Frame(si.shading_frame_ws.to_global(normal), si.shading_frame_ws.tangent);
        } else {
            mi.shading_frame_ws = si.shading_frame_ws;
        }
        mi.shading_frame_ws = stdmat::orient_shading_frame(si, mi.shading_frame_ws, double_sided);
        if (hints & MaterialInstanceHints::adjust_shading_normal) {
            adjust_shading_normal(mi.shading_frame_ws, si.front_normal_ws(), si.wi_ws);
        }

        // Resolve textured glTF metallic-roughness inputs into scalar controls.
        float3 base_color = base_color_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb * base_color_factor;

        float4 metallic_roughness = metallic_roughness_texture.sample(lod_sampler, si.uv, float4(1.0));
        float metallic = clamp(metallic_roughness[metallic_texture_channel] * metallic_factor, 0.0, 1.0);
        float roughness = clamp(metallic_roughness[roughness_texture_channel] * roughness_factor, 0.0, 1.0);

        // Convert IOR to dielectric F0, then blend toward base color for metals.
        float safe_ior = max(ior, 1e-3);
        float f = (safe_ior - 1.0) / (safe_ior + 1.0);
        float f0 = f * f;

        // Fill the common BSDF data used by both metallic-roughness and spec/gloss frontends.
        StandardBSDFData bsdf_data = {};
        bsdf_data.diffuse = base_color * (1.0 - metallic);
        bsdf_data.specular = lerp(float3(f0), base_color, metallic);
        bsdf_data.roughness = roughness;
        bsdf_data.metallic = metallic;
        bsdf_data.eta = si.front_facing ? (si.ior / safe_ior) : (safe_ior / si.ior);
        bsdf_data.transmission = transmission_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb * transmission_factor;
        bsdf_data.diffuse_transmission = saturate(diffuse_transmission_factor);
        bsdf_data.specular_transmission = saturate(specular_transmission_factor);
        bsdf_data.thin_surface = thin_surface;
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
