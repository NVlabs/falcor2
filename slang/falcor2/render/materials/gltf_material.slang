// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

// The core texture words are physically flat. A small GLTFTextureInfo accessor
// keeps sampling code shared without the D3D12 cost measured for nested fields.
public struct GLTFMaterial<let Features : uint, let Textures : uint, let Transforms : uint, let NonzeroTexcoords : uint>
    : IMaterial
{
    typedef GLTFMaterialInstance<Features> MaterialInstance;

    public static const bool HAS_ANISOTROPY = (Features & gltfmat::FeatureBits::ANISOTROPY) != 0u;
    public static const bool HAS_CLEARCOAT = (Features & gltfmat::FeatureBits::CLEARCOAT) != 0u;
    public static const bool HAS_DIFFUSE_TRANSMISSION = (Features & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u;
    public static const bool HAS_EMISSION = (Features & gltfmat::FeatureBits::EMISSION) != 0u;
    public static const bool HAS_IOR = (Features & gltfmat::FeatureBits::IOR) != 0u;
    public static const bool HAS_IRIDESCENCE = (Features & gltfmat::FeatureBits::IRIDESCENCE) != 0u;
    public static const bool HAS_NORMAL = (Features & gltfmat::FeatureBits::NORMAL) != 0u;
    public static const bool HAS_SHEEN = (Features & gltfmat::FeatureBits::SHEEN) != 0u;
    public static const bool HAS_SPECULAR = (Features & gltfmat::FeatureBits::SPECULAR) != 0u;
    public static const bool HAS_TRANSMISSION = (Features & gltfmat::FeatureBits::TRANSMISSION) != 0u;
    public static const bool HAS_VERTEX_COLOR = (Features & gltfmat::FeatureBits::VERTEX_COLOR) != 0u;
    public static const bool HAS_VOLUME = (Features & gltfmat::FeatureBits::VOLUME) != 0u;

    public MaterialHeader header;
    public TextureHandle base_color_handle;
    public uint base_color_packed;
    public TextureHandle metallic_roughness_handle;
    public uint metallic_roughness_packed;
    public TextureHandle normal_handle;
    public uint normal_packed;
    public TextureHandle emissive_handle;
    public uint emissive_packed;
    // CUDA dynamic dispatch and reflection require a padded half vector here.
    public float16_t4 base_color_factor;
    // Metallic, roughness, normal scale, reserved.
    public float16_t4 core_factors;
    public float4 emissive_factor;
    public uint feature_mask;
    public uint transform_mask;
    public BufferOffsetPair extra_data;

    private GLTFTextureInfo base_color_info()
    {
        GLTFTextureInfo info = {};
        info.handle = base_color_handle;
        info.packed = base_color_packed;
        return info;
    }

    private GLTFTextureInfo metallic_roughness_info()
    {
        GLTFTextureInfo info = {};
        info.handle = metallic_roughness_handle;
        info.packed = metallic_roughness_packed;
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

    private GLTFClearcoatRecord clearcoat_record()
    {
        return extra_data.load<GLTFClearcoatRecord>(
            gltf_material::feature_record_offset(feature_mask, gltfmat::FeatureBits::CLEARCOAT)
        );
    }

    private float
    resolve_clearcoat<LodSampler : ILodSampler>(SurfaceInteraction si, LodSampler lod_sampler, GLTFClearcoatRecord data)
    {
        const float4 sample
            = sample_texture<gltfmat::TextureBits::CLEARCOAT>(si, lod_sampler, data.texture, float4(1.0));
        return saturate(float(data.factors.x) * sample.r);
    }

    private Frame clearcoat_frame<LodSampler : ILodSampler>(
        SurfaceInteraction si,
        LodSampler lod_sampler,
        MaterialInstanceHints hints,
        GLTFClearcoatRecord data
    )
    {
        return gltf_material::
            shading_frame_sparse<Textures, Transforms, NonzeroTexcoords, gltfmat::TextureBits::CLEARCOAT_NORMAL>(
                si,
                lod_sampler,
                hints,
                header.flags,
                data.normal_texture,
                float(data.factors.z),
                feature_mask,
                transform_mask,
                extra_data
            );
    }

    // Resolve texture-dependent inputs at this hit; GLTFBSDF owns scattering.
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
        mi.shading_frame_ws
            = gltf_material::shading_frame_sparse<Textures, Transforms, NonzeroTexcoords, gltfmat::TextureBits::NORMAL>(
                si,
                lod_sampler,
                hints,
                header.flags,
                normal_texture,
                float(core_factors.z),
                feature_mask,
                transform_mask,
                extra_data
            );

        // RGB color and coverage are separate; eval_opacity() handles alpha.
        float3 base_color
            = sample_texture<gltfmat::TextureBits::BASE_COLOR>(si, lod_sampler, base_color_info(), float4(1.0)).rgb
            * base_color_factor.xyz;
        if (HAS_VERTEX_COLOR) {
            if ((feature_mask & gltfmat::FeatureBits::VERTEX_COLOR) != 0u)
                base_color *= surface_material::resolve_vertex_color(si).rgb;
        }

        // glTF metallic is B and perceptual roughness is G. The BSDF squares roughness.
        const float4 metallic_roughness = sample_texture<gltfmat::TextureBits::METALLIC_ROUGHNESS>(
            si,
            lod_sampler,
            metallic_roughness_info(),
            float4(1.0)
        );
        const float metallic = clamp(metallic_roughness.b * float(core_factors.x), 0.0, 1.0);
        const float roughness = clamp(metallic_roughness.g * float(core_factors.y), 0.0, 1.0);

        // IOR zero is the glTF infinity sentinel. A thin surface uses air outside.
        float dielectric_ior = 1.5;
        if (HAS_IOR) {
            if ((feature_mask & gltfmat::FeatureBits::IOR) != 0u) {
                const GLTFIORRecord data = extra_data.load<GLTFIORRecord>(
                    gltf_material::feature_record_offset(feature_mask, gltfmat::FeatureBits::IOR)
                );
                dielectric_ior = max(data.ior_and_reserved.x, 0.0);
            }
        }
        const bool thin_walled = is_set(header.flags, MaterialFlags::thin_walled);
        const float exterior_ior = thin_walled ? 1.0 : si.exterior_ior;
        const float f = (dielectric_ior - exterior_ior) / (dielectric_ior + exterior_ior);
        if (!thin_walled) {
            mi.interior_ior = dielectric_ior == 0.0 ? 65504.0 : dielectric_ior;
            if (HAS_VOLUME) {
                if ((feature_mask & gltfmat::FeatureBits::VOLUME) != 0u) {
                    const GLTFVolumeRecord data = extra_data.load<GLTFVolumeRecord>(
                        gltf_material::feature_record_offset(feature_mask, gltfmat::FeatureBits::VOLUME)
                    );
                    mi.volume.set(GLTFVolumeState(data.sigma_a.xyz));
                }
            }
        }

        GLTFBSDFData<Features> bsdf_data = {};
        bsdf_data.base_color = base_color;
        bsdf_data.metallic = metallic;
        bsdf_data.roughness = roughness;
        bsdf_data.ior = dielectric_ior;
        bsdf_data.thin_walled = thin_walled;
        bsdf_data.eta
            = thin_walled ? 1.0 : (si.front_facing ? exterior_ior / mi.interior_ior : mi.interior_ior / exterior_ior);

        if (HAS_TRANSMISSION) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u) {
                const GLTFTransmissionRecord data = extra_data.load<GLTFTransmissionRecord>(
                    gltf_material::feature_record_offset(feature_mask, gltfmat::FeatureBits::TRANSMISSION)
                );
                const float transmission = saturate(
                    data.factor_and_reserved.x
                    * sample_texture<gltfmat::TextureBits::TRANSMISSION>(si, lod_sampler, data.texture, float4(1.0)).r
                );
                bsdf_data.transmission.set(GLTFTransmissionInputs(transmission));
            }
        }

        // KHR_materials_specular: strength comes from A, tint from linear RGB.
        float strength = 1.0;
        float3 specular_color = float3(1.0);
        if (HAS_SPECULAR) {
            if ((feature_mask & gltfmat::FeatureBits::SPECULAR) != 0u) {
                const GLTFSpecularRecord data = extra_data.load<GLTFSpecularRecord>(
                    gltf_material::feature_record_offset(feature_mask, gltfmat::FeatureBits::SPECULAR)
                );
                strength = saturate(
                    data.color_strength.a
                    * sample_texture<gltfmat::TextureBits::SPECULAR>(si, lod_sampler, data.specular, float4(1.0)).a
                );
                specular_color = data.color_strength.rgb
                    * sample_texture<gltfmat::TextureBits::SPECULAR_COLOR>(
                          si,
                          lod_sampler,
                          data.specular_color,
                          float4(1.0)
                    )
                          .rgb;
            }
        }
        // Tint may exceed one; clamp F0 before scaling. F90 is strength.
        bsdf_data.dielectric_f0 = saturate(f * f * specular_color) * strength;
        bsdf_data.dielectric_f90 = strength;

        if (HAS_SHEEN) {
            if ((feature_mask & gltfmat::FeatureBits::SHEEN) != 0u) {
                const GLTFSheenRecord data = extra_data.load<GLTFSheenRecord>(
                    gltf_material::feature_record_offset(feature_mask, gltfmat::FeatureBits::SHEEN)
                );
                const float3 sheen_color = saturate(
                    data.color_roughness.rgb
                    * sample_texture<gltfmat::TextureBits::SHEEN_COLOR>(
                          si,
                          lod_sampler,
                          data.color_texture,
                          float4(1.0)
                    )
                          .rgb
                );
                const float sheen_roughness = saturate(
                    data.color_roughness.a
                    * sample_texture<gltfmat::TextureBits::SHEEN_ROUGHNESS>(
                          si,
                          lod_sampler,
                          data.roughness_texture,
                          float4(1.0)
                    )
                          .a
                );
                bsdf_data.sheen.set(GLTFSheenInputs(sheen_color, sheen_roughness));
            }
        }

        if (HAS_IRIDESCENCE) {
            if ((feature_mask & gltfmat::FeatureBits::IRIDESCENCE) != 0u) {
                const GLTFIridescenceRecord data = extra_data.load<GLTFIridescenceRecord>(
                    gltf_material::feature_record_offset(feature_mask, gltfmat::FeatureBits::IRIDESCENCE)
                );
                const float4 iridescence
                    = sample_texture<gltfmat::TextureBits::IRIDESCENCE>(si, lod_sampler, data.texture, float4(1.0));
                const float4 thickness = sample_texture<gltfmat::TextureBits::IRIDESCENCE_THICKNESS>(
                    si,
                    lod_sampler,
                    data.thickness_texture,
                    float4(1.0)
                );
                const float iridescence_factor = saturate(data.factor_ior_thickness.x * iridescence.r);
                const float iridescence_ior = max(data.factor_ior_thickness.y, 1.0);
                // G selects nanometers between the authored min and max.
                const float iridescence_thickness
                    = max(0.0, lerp(data.factor_ior_thickness.z, data.factor_ior_thickness.w, thickness.g));
                bsdf_data.iridescence.set(
                    GLTFIridescenceInputs(iridescence_factor, iridescence_ior, iridescence_thickness)
                );
            }
        }

        if (HAS_ANISOTROPY) {
            if ((feature_mask & gltfmat::FeatureBits::ANISOTROPY) != 0u) {
                const GLTFAnisotropyRecord data = extra_data.load<GLTFAnisotropyRecord>(
                    gltf_material::feature_record_offset(feature_mask, gltfmat::FeatureBits::ANISOTROPY)
                );
                const float4 sample = sample_texture<gltfmat::TextureBits::ANISOTROPY>(
                    si,
                    lod_sampler,
                    data.texture,
                    float4(1.0, 0.5, 1.0, 1.0)
                );
                const float anisotropy_strength = saturate(data.strength_rotation.x * sample.b);
                float2 direction = sample.rg * 2.0 - 1.0;
                direction = dot(direction, direction) > 0.0 ? normalize(direction) : float2(1.0, 0.0);
                const float c = cos(data.strength_rotation.y);
                const float s = sin(data.strength_rotation.y);
                const float2 anisotropy_direction
                    = float2(c * direction.x - s * direction.y, s * direction.x + c * direction.y);
                bsdf_data.anisotropy.set(GLTFAnisotropyInputs(anisotropy_strength, anisotropy_direction));
            }
        }

        if (HAS_DIFFUSE_TRANSMISSION) {
            if ((feature_mask & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u) {
                const GLTFDiffuseTransmissionRecord data = extra_data.load<GLTFDiffuseTransmissionRecord>(
                    gltf_material::feature_record_offset(feature_mask, gltfmat::FeatureBits::DIFFUSE_TRANSMISSION)
                );
                const float diffuse_transmission = saturate(
                    data.color_factor.a
                    * sample_texture<gltfmat::TextureBits::DIFFUSE_TRANSMISSION>(
                          si,
                          lod_sampler,
                          data.texture,
                          float4(1.0)
                    )
                          .a
                );
                const float3 diffuse_transmission_color = saturate(
                    data.color_factor.rgb
                    * sample_texture<gltfmat::TextureBits::DIFFUSE_TRANSMISSION_COLOR>(
                          si,
                          lod_sampler,
                          data.color_texture,
                          float4(1.0)
                    )
                          .rgb
                );
                bsdf_data.diffuse_transmission.set(
                    GLTFDiffuseTransmissionInputs(diffuse_transmission, diffuse_transmission_color)
                );
            }
        }
        mi.bsdf_data = bsdf_data;

        // Emission ignores scattering hints. The value occupies no instance
        // storage when the compile-time feature bit is absent.
        if (HAS_EMISSION) {
            if ((feature_mask & gltfmat::FeatureBits::EMISSION) != 0u)
                mi.emission.set(eval_emissive_radiance(si, lod_sampler));
        }

        if (HAS_CLEARCOAT) {
            if ((feature_mask & gltfmat::FeatureBits::CLEARCOAT) != 0u) {
                const GLTFClearcoatRecord data = clearcoat_record();
                const float factor = resolve_clearcoat(si, lod_sampler, data);
                if (factor > 0.0) {
                    const Frame frame_ws = clearcoat_frame(si, lod_sampler, hints, data);
                    const float roughness = saturate(
                        float(data.factors.y)
                        * sample_texture<gltfmat::TextureBits::CLEARCOAT_ROUGHNESS>(
                              si,
                              lod_sampler,
                              data.roughness_texture,
                              float4(1.0)
                        )
                              .g
                    );
                    mi.clearcoat.set(GLTFClearcoatState(frame_ws, factor, roughness));
                }
            }
        }
        return mi;
    }

    public override bool is_emissive()
    {
        if (HAS_EMISSION) {
            if ((feature_mask & gltfmat::FeatureBits::EMISSION) != 0u)
                return any(emissive_factor.xyz > 0.0);
        }
        return false;
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
        float result = header.opacity_factor;
        result *= sample_texture<gltfmat::TextureBits::BASE_COLOR>(si, lod_sampler, base_color_info(), float4(1.0)).a;
        if (HAS_VERTEX_COLOR) {
            if ((feature_mask & gltfmat::FeatureBits::VERTEX_COLOR) != 0u)
                result *= surface_material::resolve_vertex_color(si).a;
        }
        return result;
    }

    public override uint2 compute_emission_sampling_grid(float2 uv[3]) { return uint2(100, 100); }

    public override float3 eval_emission<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        // Emitter selection uses the uncoated power estimate.
        return uncoated_emissive_radiance(si, lod_sampler) * float.PI;
    }

    public override float3 eval_emissive_radiance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        if (!HAS_EMISSION)
            return float3(0.0);
        if ((feature_mask & gltfmat::FeatureBits::EMISSION) == 0u)
            return float3(0.0);
        float3 emission = uncoated_emissive_radiance(si, lod_sampler);
        if (HAS_CLEARCOAT) {
            if ((feature_mask & gltfmat::FeatureBits::CLEARCOAT) != 0u) {
                const GLTFClearcoatRecord data = clearcoat_record();
                const float factor = resolve_clearcoat(si, lod_sampler, data);
                if (factor > 0.0) {
                    const Frame frame = clearcoat_frame(si, lod_sampler, hints, data);
                    // Coat attenuation applies to both direct and BSDF-hit emission.
                    emission *= 1.0 - gltf_material::clearcoat_weight(factor, dot(si.wi_ws, frame.normal));
                }
            }
        }
        return emission;
    }

    private float3
    uncoated_emissive_radiance<LodSampler : ILodSampler>(const SurfaceInteraction si, const LodSampler lod_sampler)
    {
        if (HAS_EMISSION) {
            if ((feature_mask & gltfmat::FeatureBits::EMISSION) != 0u) {
                return sample_texture<gltfmat::TextureBits::EMISSIVE>(si, lod_sampler, emissive_info(), float4(1.0)).rgb
                    * emissive_factor.xyz;
            }
        }
        return float3(0.0);
    }
};
