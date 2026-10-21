// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct GLTFMaterial : IMaterial
{
    typedef GLTFMaterialInstance MaterialInstance;

    public MaterialHeader header;
    public TextureHandle base_color_texture;
    public TextureHandle normal_texture;
    public TextureHandle metallic_roughness_texture;
    public TextureHandle emissive_texture;
    // TODO: These should be float16_t3 but CUDA backend has inconsistent reflection and dynamic dispatch layout rules!
    public float16_t4 base_color_factor;
    public float16_t metallic_factor;
    public float16_t roughness_factor;
    public float16_t normal_texture_scale;
    public float16_t ior;
    public uint texture_coordinate_selectors;
    public float3 emissive_factor;
    public uint transform_mask;
    public uint transform_offset;
    public BufferOffsetPair texture_transform_data;
    public TextureHandle specular_texture;
    public TextureHandle specular_color_texture;
    public float16_t4 specular_color_factor;
    public float16_t specular_factor;
    public float16_t transmission_factor;
    public TextureHandle transmission_texture;
    public float16_t clearcoat_factor;
    public float16_t clearcoat_roughness_factor;
    public uint clearcoat_offset;
    public uint sheen_offset = 0xffffffffu;
    public uint iridescence_offset = 0xffffffffu;

    private float resolve_clearcoat<L : ILodSampler>(SurfaceInteraction si, L lod_sampler, GLTFClearcoatData data)
    {
        float2 uv = gltf_material::resolve_texcoord(
            si,
            gltfmat::ClearcoatTCBits::texture::extract(data.selectors),
            7,
            transform_mask,
            texture_transform_data,
            transform_offset
        );
        return saturate(float(clearcoat_factor) * data.texture.sample(lod_sampler, uv, float4(1.0)).r);
    }

    private Frame clearcoat_frame<L : ILodSampler>(
        SurfaceInteraction si,
        L lod_sampler,
        MaterialInstanceHints hints,
        GLTFClearcoatData data
    )
    {
        return gltf_material::shading_frame(
            si,
            lod_sampler,
            hints,
            header.flags,
            data.normal_texture,
            data.normal_scale,
            gltfmat::ClearcoatTCBits::normal::extract(data.selectors),
            transform_mask,
            texture_transform_data,
            transform_offset,
            9,
            gltfmat::ClearcoatTCBits::normal_map_encoded::extract(data.selectors) != 0
        );
    }

    // Resolve texture-dependent parameters at this hit; GLTFBSDF evaluates and samples their scattering.
    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        // Absent volumes and coatings retain unit interior IOR, zero absorption, and no clearcoat.
        MaterialInstance mi = {};

        // Map tangent-space normals into world space before preparing any direction-dependent lobes.
        // Binding indices address UV transforms; coordinate selectors choose the mesh UV set.
        mi.shading_frame_ws = gltf_material::shading_frame(
            si,
            lod_sampler,
            hints,
            header.flags,
            normal_texture,
            float(normal_texture_scale),
            gltfmat::TCBits::normal::extract(texture_coordinate_selectors),
            transform_mask,
            texture_transform_data,
            transform_offset,
            2,
            gltfmat::TCBits::normal_map_encoded::extract(texture_coordinate_selectors) != 0
        );

        // Color textures are sampled in linear space. White fallbacks preserve the authored factors.
        // Alpha coverage is evaluated separately by eval_opacity().
        // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material
        float3 base_color;
        {
            const float2 texture_coord = gltf_material::resolve_texcoord(
                si,
                gltfmat::TCBits::base_color::extract(texture_coordinate_selectors),
                0,
                transform_mask,
                texture_transform_data,
                transform_offset
            );
            base_color = base_color_texture.sample(lod_sampler, texture_coord, float4(1.0)).rgb * base_color_factor.xyz;
            if (gltfmat::TCBits::use_vertex_color::extract(texture_coordinate_selectors) != 0)
                base_color *= surface_material::resolve_vertex_color(si).rgb;
        }

        float4 metallic_roughness;
        {
            const float2 texture_coord = gltf_material::resolve_texcoord(
                si,
                gltfmat::TCBits::metallic_roughness::extract(texture_coordinate_selectors),
                1,
                transform_mask,
                texture_transform_data,
                transform_offset
            );
            metallic_roughness = metallic_roughness_texture.sample(lod_sampler, texture_coord, float4(1.0));
        }

        // glTF packs metallic in B and perceptual roughness in G. Keep r here: GLTFBSDF uses alpha = r*r.
        // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material
        float metallic = clamp(metallic_roughness.b * metallic_factor, 0.0, 1.0);
        float roughness = clamp(metallic_roughness.g * roughness_factor, 0.0, 1.0);

        // Normal-incidence dielectric reflectance is F0 = ((n_material - n_outside)/(n_material + n_outside))^2.
        // Thin sheets use air here; solid boundaries use the tracer's tracked exterior medium.
        float dielectric_ior = max(float(ior), 0.0);
        bool thin_walled = is_set(header.flags, MaterialFlags::thin_walled);
        float exterior_ior = thin_walled ? 1.0 : si.exterior_ior;

        // For glTF's IOR-zero sentinel this ratio is -1, giving the required unit untinted F0.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_ior/README.md
        float f = (dielectric_ior - exterior_ior) / (dielectric_ior + exterior_ior);
        if (!thin_walled) {

            // IOR zero denotes infinity; the interior list stores binary16 IOR.
            mi.interior_ior = dielectric_ior == 0.0 ? 65504.0 : dielectric_ior;

            // The CPU stores sigma_a = -log(attenuation_color)/attenuation_distance for Beer-Lambert transport.
            // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_volume/README.md
            mi.volume_sigma_a = texture_transform_data.load<GLTFVolumeData>(0).sigma_a;
        }
        GLTFBSDFData bsdf_data = {};
        bsdf_data.base_color = base_color;
        bsdf_data.metallic = metallic;
        bsdf_data.roughness = roughness;
        bsdf_data.ior = dielectric_ior;
        bsdf_data.thin_walled = thin_walled;

        // Refraction uses n_incident/n_transmitted: reverse the ratio when leaving a solid.
        // Thin-sheet transmission has its own lobe and does not use this refraction ratio.
        bsdf_data.eta
            = thin_walled ? 1.0 : (si.front_facing ? exterior_ior / mi.interior_ior : mi.interior_ior / exterior_ior);

        // Renderer policy: the scene binds the lookup table used for GGX metal energy compensation.
        bsdf_data.compensate_metal = true;

        // KHR_materials_transmission uses R. The BSDF applies the nonmetal weight and Fresnel split later.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md
        bsdf_data.transmission = saturate(
            float(transmission_factor)
            * transmission_texture
                  .sample(
                      lod_sampler,
                      gltf_material::resolve_texcoord(
                          si,
                          gltfmat::TCBits::transmission::extract(texture_coordinate_selectors),
                          6,
                          transform_mask,
                          texture_transform_data,
                          transform_offset
                      ),
                      float4(1.0)
                  )
                  .r
        );

        // KHR_materials_specular stores scalar strength in A and RGB tint in a separate color texture.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md
        float strength = saturate(
            float(specular_factor)
            * specular_texture
                  .sample(
                      lod_sampler,
                      gltf_material::resolve_texcoord(
                          si,
                          gltfmat::TCBits::specular::extract(texture_coordinate_selectors),
                          4,
                          transform_mask,
                          texture_transform_data,
                          transform_offset
                      ),
                      float4(1.0)
                  )
                  .a
        );
        float3 specular_color = specular_color_factor.xyz
            * specular_color_texture
                  .sample(
                      lod_sampler,
                      gltf_material::resolve_texcoord(
                          si,
                          gltfmat::TCBits::specular_color::extract(texture_coordinate_selectors),
                          5,
                          transform_mask,
                          texture_transform_data,
                          transform_offset
                      ),
                      float4(1.0)
                  )
                  .rgb;

        // Clamp tinted F0 before scaling by strength; tint may exceed one. F90 is strength, not one.
        // This preserves the extension's weighted Schlick curve at both normal and grazing incidence.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md#implementation
        bsdf_data.dielectric_f0 = saturate(f * f * specular_color) * strength;
        bsdf_data.dielectric_f90 = strength;

        // Sheen has independent RGB color and A roughness; its BRDF and base-layer attenuation are deferred.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_sheen/README.md#sheen
        if (sheen_offset != 0xffffffffu) {
            const GLTFSheenData data = texture_transform_data.load<GLTFSheenData>(sheen_offset);
            const float4 sheen_color_sample = data.color_texture.sample(
                lod_sampler,
                gltf_material::resolve_texcoord(
                    si,
                    gltfmat::SheenTCBits::color::extract(data.selectors),
                    10,
                    transform_mask,
                    texture_transform_data,
                    transform_offset
                ),
                float4(1.0)
            );
            const float4 sheen_roughness_sample = data.roughness_texture.sample(
                lod_sampler,
                gltf_material::resolve_texcoord(
                    si,
                    gltfmat::SheenTCBits::roughness::extract(data.selectors),
                    11,
                    transform_mask,
                    texture_transform_data,
                    transform_offset
                ),
                float4(1.0)
            );
            bsdf_data.sheen_color = saturate(data.color * sheen_color_sample.rgb);
            bsdf_data.sheen_roughness = saturate(data.roughness * sheen_roughness_sample.a);
        }

        // Resolve thin-film inputs only; the iridescence Fresnel calculation is performed in GLTFBSDF.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_iridescence/README.md
        if (iridescence_offset != 0xffffffffu) {
            const GLTFIridescenceData data = texture_transform_data.load<GLTFIridescenceData>(iridescence_offset);
            const float4 iridescence_sample = data.texture.sample(
                lod_sampler,
                gltf_material::resolve_texcoord(
                    si,
                    gltfmat::IridescenceTCBits::texture::extract(data.selectors),
                    12,
                    transform_mask,
                    texture_transform_data,
                    transform_offset
                ),
                float4(1.0)
            );
            const float4 iridescence_thickness_sample = data.thickness_texture.sample(
                lod_sampler,
                gltf_material::resolve_texcoord(
                    si,
                    gltfmat::IridescenceTCBits::thickness::extract(data.selectors),
                    13,
                    transform_mask,
                    texture_transform_data,
                    transform_offset
                ),
                float4(1.0)
            );
            bsdf_data.iridescence = saturate(data.factor * iridescence_sample.r);
            bsdf_data.iridescence_ior = max(data.ior, 1.0);

            // G interpolates thickness in nanometers. A missing texture samples one, selecting the maximum.
            bsdf_data.iridescence_thickness
                = max(0.0, lerp(data.thickness_minimum, data.thickness_maximum, iridescence_thickness_sample.g));
        }

        // This optional record is stored immediately before the sparse UV-transform records.
        if (gltfmat::TCBits::has_surface_extensions::extract(texture_coordinate_selectors) != 0) {
            const GLTFSurfaceExtensionsData data = texture_transform_data.load<GLTFSurfaceExtensionsData>(
                transform_offset - GLTF_SURFACE_EXTENSIONS_SIZE
            );
            const float4 anisotropy_sample = data.anisotropy_texture.sample(
                lod_sampler,
                gltf_material::resolve_texcoord(
                    si,
                    gltfmat::SurfaceExtensionsTCBits::anisotropy::extract(data.selectors),
                    14,
                    transform_mask,
                    texture_transform_data,
                    transform_offset
                ),
                float4(1.0, 0.5, 1.0, 1.0)
            );
            const float4 diffuse_transmission_sample = data.diffuse_transmission_texture.sample(
                lod_sampler,
                gltf_material::resolve_texcoord(
                    si,
                    gltfmat::SurfaceExtensionsTCBits::diffuse_transmission::extract(data.selectors),
                    15,
                    transform_mask,
                    texture_transform_data,
                    transform_offset
                ),
                float4(1.0)
            );
            const float4 diffuse_transmission_color_sample = data.diffuse_transmission_color_texture.sample(
                lod_sampler,
                gltf_material::resolve_texcoord(
                    si,
                    gltfmat::SurfaceExtensionsTCBits::diffuse_transmission_color::extract(data.selectors),
                    16,
                    transform_mask,
                    texture_transform_data,
                    transform_offset
                ),
                float4(1.0)
            );

            // B modulates strength; RG encodes a signed direction in the shading tangent/bitangent plane.
            // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_anisotropy/README.md
            bsdf_data.anisotropy_strength = saturate(data.anisotropy_strength * anisotropy_sample.b);
            float2 direction = anisotropy_sample.rg * 2.0 - 1.0;

            // A zero direction cannot be normalized; use the tangent axis as a deterministic fallback.
            direction = dot(direction, direction) > 0.0 ? normalize(direction) : float2(1.0, 0.0);

            // Apply the authored angle in radians with the 2D rotation [c -s; s c].
            float c = cos(data.anisotropy_rotation), s = sin(data.anisotropy_rotation);
            bsdf_data.anisotropy_direction
                = float2(c * direction.x - s * direction.y, s * direction.x + c * direction.y);

            // Diffuse transmission uses A for weight and linear RGB for tint; the BSDF splits the diffuse share.
            // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_diffuse_transmission/README.md
            bsdf_data.diffuse_transmission = saturate(data.diffuse_transmission_factor * diffuse_transmission_sample.a);
            bsdf_data.diffuse_transmission_color
                = saturate(data.diffuse_transmission_color_factor * diffuse_transmission_color_sample.rgb);
        }
        mi.bsdf_data = bsdf_data;

        // Match direct-light evaluation: scattering hints do not affect emission.
        mi.emissive_radiance = eval_emissive_radiance(si, lod_sampler);

        // Clearcoat has its own normal and perceptual roughness, separate from the base layer.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_clearcoat/README.md#clearcoat
        if (clearcoat_factor > 0.0) {
            const GLTFClearcoatData data = texture_transform_data.load<GLTFClearcoatData>(clearcoat_offset);
            mi.clearcoat = resolve_clearcoat(si, lod_sampler, data);
            if (mi.clearcoat > 0.0) {
                mi.clearcoat_frame_ws = clearcoat_frame(si, lod_sampler, hints, data);
                const float2 uv = gltf_material::resolve_texcoord(
                    si,
                    gltfmat::ClearcoatTCBits::roughness::extract(data.selectors),
                    8,
                    transform_mask,
                    texture_transform_data,
                    transform_offset
                );
                mi.clearcoat_roughness = saturate(
                    float(clearcoat_roughness_factor) * data.roughness_texture.sample(lod_sampler, uv, float4(1.0)).g
                );
            }
        }

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
        return gltf_material::opacity(
            si,
            lod_sampler,
            header,
            base_color_texture,
            gltfmat::TCBits::base_color::extract(texture_coordinate_selectors),
            gltfmat::TCBits::use_vertex_color::extract(texture_coordinate_selectors) != 0,
            transform_mask,
            texture_transform_data,
            transform_offset
        );
    }

    public override uint2 compute_emission_sampling_grid(float2 uv[3]) { return uint2(100, 100); }

    public override float3 eval_emission<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        // Conservative power estimate for emitter selection. Directional radiance
        // below applies the coat consistently for NEE and BSDF-hit emission.
        return uncoated_emissive_radiance(si, lod_sampler) * float.PI;
    }

    public override float3 eval_emissive_radiance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        float3 emission = uncoated_emissive_radiance(si, lod_sampler);
        if (clearcoat_factor > 0.0) {
            const GLTFClearcoatData data = texture_transform_data.load<GLTFClearcoatData>(clearcoat_offset);
            const float factor = resolve_clearcoat(si, lod_sampler, data);
            if (factor > 0.0) {
                const Frame frame = clearcoat_frame(si, lod_sampler, hints, data);
                // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_clearcoat/README.md#emission
                emission *= 1.0 - gltf_material::clearcoat_weight(factor, dot(si.wi_ws, frame.normal));
            }
        }
        return emission;
    }

    private float3
    uncoated_emissive_radiance<LodSampler : ILodSampler>(const SurfaceInteraction si, const LodSampler lod_sampler)
    {
        return gltf_material::emission(
            si,
            lod_sampler,
            emissive_texture,
            emissive_factor,
            gltfmat::TCBits::emissive::extract(texture_coordinate_selectors),
            transform_mask,
            texture_transform_data,
            transform_offset
        );
    }
};
