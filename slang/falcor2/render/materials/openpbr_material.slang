// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

#define OPENPBR_LANGUAGE_TARGET_SLANG 1
#define OPENPBR_UINT16 uint16_t
#define OPENPBR_USE_TEXTURE_LUTS 1
#define OPENPBR_SAMPLE_2D_TEXTURE(lut_id, uv) openpbr_sample_2d_texture(lut_id, uv)
#define OPENPBR_SAMPLE_3D_TEXTURE(lut_id, uv) openpbr_sample_3d_texture(lut_id, uv)

#include "openpbr-bsdf/openpbr.h"

public struct OpenPBRGlobals
{
    DescriptorHandle<Texture3D> OpenPBR_IdealDielectricEnergyComplement;
    DescriptorHandle<Texture2D> OpenPBR_IdealDielectricAverageEnergyComplement;
    DescriptorHandle<Texture2D> OpenPBR_IdealDielectricReflectionRatio;
    DescriptorHandle<Texture3D> OpenPBR_OpaqueDielectricEnergyComplement;
    DescriptorHandle<Texture2D> OpenPBR_OpaqueDielectricAverageEnergyComplement;
    DescriptorHandle<Texture2D> OpenPBR_IdealMetalEnergyComplement;
    DescriptorHandle<Texture2D> OpenPBR_IdealMetalAverageEnergyComplement;
    DescriptorHandle<Texture2D> OpenPBR_LTC;
    DescriptorHandle<SamplerState> sampler;
};

public OpenPBRGlobals openpbr_globals;

internal float4 openpbr_sample_2d_texture(const OpenPBR_LutId lut_id, float2 uv)
{
    // clang-format off
    switch (lut_id) {
    case OpenPBR_LutId_IdealDielectricAverageEnergyComplement:
        return openpbr_globals.OpenPBR_IdealDielectricAverageEnergyComplement.SampleLevel(openpbr_globals.sampler, uv, 0.0);
    case OpenPBR_LutId_IdealDielectricReflectionRatio:
        return openpbr_globals.OpenPBR_IdealDielectricReflectionRatio.SampleLevel(openpbr_globals.sampler, uv, 0.0);
    case OpenPBR_LutId_OpaqueDielectricAverageEnergyComplement:
        return openpbr_globals.OpenPBR_OpaqueDielectricAverageEnergyComplement.SampleLevel(openpbr_globals.sampler, uv, 0.0);
    case OpenPBR_LutId_IdealMetalEnergyComplement:
        return openpbr_globals.OpenPBR_IdealMetalEnergyComplement.SampleLevel(openpbr_globals.sampler, uv, 0.0);
    case OpenPBR_LutId_IdealMetalAverageEnergyComplement:
        return openpbr_globals.OpenPBR_IdealMetalAverageEnergyComplement.SampleLevel(openpbr_globals.sampler, uv, 0.0);
    case OpenPBR_LutId_LTC:
        return openpbr_globals.OpenPBR_LTC.SampleLevel(openpbr_globals.sampler, uv, 0.0);
    }
    // clang-format on
    return float4(0);
}

internal float4 openpbr_sample_3d_texture(const OpenPBR_LutId lut_id, float3 uv)
{
    // clang-format off
    switch (lut_id) {
    case OpenPBR_LutId_IdealDielectricEnergyComplement:
        return openpbr_globals.OpenPBR_IdealDielectricEnergyComplement.SampleLevel(openpbr_globals.sampler, uv, 0.0);
    case OpenPBR_LutId_OpaqueDielectricEnergyComplement:
        return openpbr_globals.OpenPBR_OpaqueDielectricEnergyComplement.SampleLevel(openpbr_globals.sampler, uv, 0.0);
    }
    // clang-format on
    return float4(0);
}

/// Compute MaterialProperties from OpenPBR resolved inputs.
///
/// This extracts approximate directional-hemispherical albedos from the internal lobe hierarchy
/// of the prepared OpenPBR BSDF.
///
/// @param inputs Fully resolved OpenPBR material inputs.
/// @param wi_ws Incident direction in world-space (pointing away from the surface).
/// @param emissive_radiance Pre-computed emissive radiance (emission_luminance * emission_color).
/// @return Populated MaterialProperties with albedo decomposition, roughness, normal, and flags.
internal MaterialProperties
openpbr_collect_properties(const OpenPBR_ResolvedInputs inputs, const float3 wi_ws, const float3 emissive_radiance)
{
    // Prepare the full BSDF with unit path throughput.
    const OpenPBR_PreparedBsdf bsdf
        = openpbr_prepare(inputs, float3(1.0), OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, wi_ws);

    // Shortcuts into the lobe hierarchy: fuzz -> coat -> base (aggregate of 6 lobes).
    let fuzz = bsdf.fuzz_lobe;
    let coat = fuzz.coating_lobe;
    let base = coat.base_lobe;

    // Fuzz pass-through: fraction of light NOT reflected by the fuzz layer.
    float fuzz_pass = openpbr_base_layer_scale_incoming(fuzz);
    // Coat pass-through (vec3): includes coat Fresnel transmission, tinting, and darkening.
    float3 coat_pass = coat.in_base_layer_scale;
    // Combined attenuation for base lobes.
    float3 base_atten = coat_pass * fuzz_pass;

    // Specular reflection/transmission coefficients from the base specular lobe.
    float cos_theta = abs(dot(wi_ws, inputs.geometry_basis.n));
    OpenPBR_AllCoefficients coeffs = openpbr_all_coefficients(base.specular_lobe.refl_trans_coeff, cos_theta);

    // Compute diffuse_reflection_albedo.
    // Diffuse lobe contribution (energy-compensated for specular layer above it).
    float3 diffuse_reflection_albedo
        = base.diffuse_lobe.diffuse_albedo * base.diffuse_lobe.cached_specular_energy_compensation * base_atten;
    // Fuzz contributes a wide, diffuse-like reflection.
    diffuse_reflection_albedo += fuzz.view_reflected * fuzz.tint;

    // Compute specular_reflection_albedo.
    // Base specular reflection contribution.
    float3 specular_reflection_albedo = coeffs.reflection_coefficient * base_atten;
    // Coat specular reflection contribution.
    specular_reflection_albedo += coat.in_reflected * fuzz_pass;

    // Compute specular_transmission_albedo.
    float3 specular_transmission_albedo = coeffs.transmission_coefficient * base_atten;
    // Thin-wall specular transmission: the flipped lobe uses a constant reflection coefficient
    // that stores the transmission color.
    specular_transmission_albedo += base.thin_wall_specular_trans_lobe.flipped_lobe.refl_trans_coeff.color * base_atten;

    // Compute diffuse_transmission_albedo.
    float3 diffuse_transmission_albedo = base.thin_wall_diffuse_trans_lobe.flipped_lobe.diffuse_albedo
        * base.thin_wall_diffuse_trans_lobe.flipped_lobe.cached_specular_energy_compensation * base_atten;

    // Compute specular reflectance at normal incidence (F0).
    float ior = inputs.specular_ior;
    float f0_dielectric = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0));
    float3 dielectric_F0 = float3(f0_dielectric) * inputs.specular_weight * inputs.specular_color;
    float3 metal_F0 = inputs.base_color * inputs.base_weight;
    float3 specular_reflectance = lerp(dielectric_F0, metal_F0, inputs.base_metalness);

    // Assemble result.
    MaterialProperties properties = {};
    properties.emission = emissive_radiance;
    properties.roughness = inputs.specular_roughness;
    properties.guide_normal = inputs.geometry_basis.n;
    properties.diffuse_reflection_albedo = diffuse_reflection_albedo;
    properties.diffuse_transmission_albedo = diffuse_transmission_albedo;
    properties.specular_reflection_albedo = specular_reflection_albedo;
    properties.specular_transmission_albedo = specular_transmission_albedo;
    properties.specular_reflectance = specular_reflectance;
    if (inputs.transmission_weight > 0.0 || inputs.subsurface_weight > 0.0)
        properties.flags = properties.flags | MaterialProperties::Flags::is_transmissive;
    if (inputs.geometry_thin_walled)
        properties.flags = properties.flags | MaterialProperties::Flags::is_thin_walled;

    return properties;
}

public struct OpenPBRMaterialInstance : IMaterialInstance
{
    OpenPBR_ResolvedInputs inputs = {};
    float3 emissive_radiance = {};

    internal OpenPBR_PreparedBsdf prepare_bsdf(const SurfaceInteraction si)
    {
        return openpbr_prepare(inputs, float3(1.0), OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, si.wi_ws);
    }

    public float3 eval<S : ISampleGenerator>(const SurfaceInteraction si, const float3 wo_ws, inout S sg)
    {
        const OpenPBR_PreparedBsdf bsdf = prepare_bsdf(si);
        OpenPBR_DiffuseSpecular result = openpbr_eval(bsdf, wo_ws);
        return result.diffuse + result.specular;
    }

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out BSDFSample result)
    {
        result = {};

        const OpenPBR_PreparedBsdf bsdf = prepare_bsdf(si);
        OpenPBR_DiffuseSpecular weight = {};
        OpenPBR_BsdfLobeType sampled_type = OpenPBR_BsdfLobeTypeNone;
        openpbr_sample(bsdf, next_float3(sg), result.wo_ws, weight, result.pdf, sampled_type);
        if (result.pdf <= 0.0)
            return false;

        result.weight = weight.diffuse + weight.specular;
        uint lobe_types = 0;
        if ((sampled_type & OpenPBR_BsdfLobeTypeReflection) != 0)
            lobe_types |= uint(LobeTypes::reflection);
        if ((sampled_type & OpenPBR_BsdfLobeTypeTransmission) != 0)
            lobe_types |= uint(LobeTypes::transmission);
        if ((sampled_type & OpenPBR_BsdfLobeTypeDiffuse) != 0)
            lobe_types |= uint(LobeTypes::diffuse);
        if ((sampled_type & OpenPBR_BsdfLobeTypeGlossy) != 0)
            lobe_types |= uint(LobeTypes::glossy);
        if ((sampled_type & OpenPBR_BsdfLobeTypeSpecular) != 0)
            lobe_types |= uint(LobeTypes::delta);
        // TODO: figure out how to classify volume scattering lobes, if we want to support them in the future
        // if ((sampled_type & OpenPBR_BsdfLobeTypeVolume) != 0)
        //     lobe_types |= 0;
        result.lobe_types = LobeTypes(lobe_types);

        return true;
    }

    public float eval_pdf(const SurfaceInteraction si, const float3 wo_ws)
    {
        const OpenPBR_PreparedBsdf bsdf = prepare_bsdf(si);
        return openpbr_pdf(bsdf, wo_ws);
    }

    public override bool is_emissive() { return any(emissive_radiance > 0.0); }

    public override float3 eval_emissive_radiance() { return emissive_radiance; }

    public override bool is_thin_walled() { return inputs.geometry_thin_walled; }

    public override MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        return openpbr_collect_properties(inputs, si.wi_ws, emissive_radiance);
    }

    public override Optional<ExtraBsdfProperties>
    collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        ExtraBsdfProperties result = {};

        result.bsdf_count = 1;
        result.bsdf_N[0] = si.shading_frame_ws.to_local(inputs.geometry_basis.n);
        result.bsdf_T[0] = si.shading_frame_ws.to_local(inputs.geometry_basis.t);
        result.bsdf_B[0] = si.shading_frame_ws.to_local(inputs.geometry_basis.b);
        result.bsdf_albedo[0] = inputs.base_color * inputs.base_weight;
        result.bsdf_weight[0] = float3(1.0);
        result.bsdf_roughness[0] = float2(inputs.specular_roughness);

        return result;
    }

    public override LobeTypes get_lobe_types(const SurfaceInteraction si)
    {
        // TODO: Return a more accurate set of lobes.
        return LobeTypes::all;
    }
};

public struct OpenPBRMaterial : IMaterial
{
    typedef OpenPBRMaterialInstance MaterialInstance;

    public struct FloatAttribute
    {
        float16_t value;
        uint16_t texture_channel;
        TextureHandle texture_handle;
    };
    public struct Float3Attribute
    {
        float16_t3 value;
        uint16_t _pad0;
        TextureHandle texture_handle;
    };

    public struct Attributes
    {
        // Base
        FloatAttribute base_weight;
        Float3Attribute base_color;
        FloatAttribute base_diffuse_roughness;
        FloatAttribute base_metalness;
        // Specular
        FloatAttribute specular_weight;
        Float3Attribute specular_color;
        FloatAttribute specular_roughness;
        FloatAttribute specular_roughness_anisotropy;
        FloatAttribute specular_ior;
        // Coat
        FloatAttribute coat_weight;
        Float3Attribute coat_color;
        FloatAttribute coat_roughness;
        FloatAttribute coat_roughness_anisotropy;
        FloatAttribute coat_ior;
        FloatAttribute coat_darkening;
        // Fuzz
        FloatAttribute fuzz_weight;
        Float3Attribute fuzz_color;
        FloatAttribute fuzz_roughness;
        // Transmission / volume
        FloatAttribute transmission_weight;
        Float3Attribute transmission_color;
        FloatAttribute transmission_depth;
        Float3Attribute transmission_scatter;
        FloatAttribute transmission_scatter_anisotropy;
        FloatAttribute transmission_dispersion_scale;
        FloatAttribute transmission_dispersion_abbe_number;
        // Thin-film
        FloatAttribute thin_film_weight;
        FloatAttribute thin_film_thickness;
        FloatAttribute thin_film_ior;
        // Emission
        FloatAttribute emission_luminance;
        Float3Attribute emission_color;
    };

    public MaterialHeader header;
    public TextureHandle normal_texture_handle;
    public float normal_texture_scale;
    public bool has_emission;
    public bool geometry_thin_walled;
    public BufferHandle attributes_buffer;

    // Attributes attributes;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};

        // Setup shading frame.
        Frame shading_frame_ws = {};
        if (normal_texture_handle.is_valid()) {
            float3 normal = normalize(
                normal_texture_handle.sample(lod_sampler, si.uv, float4(0.0, 0.0, 1.0, 0.0)).xyz * 2.0 - 1.0
            );
            normal = normalize(lerp(float3(0.0, 0.0, 1.0), normal, normal_texture_scale));
            shading_frame_ws = Frame(
                si.shading_frame_ws.to_global(normal),
                si.shading_frame_ws.tangent,
                si.shading_frame_ws.handedness_sign()
            );
        } else {
            shading_frame_ws = si.shading_frame_ws;
        }
        if (hints & MaterialInstanceHints::adjust_shading_normal) {
            adjust_shading_normal(shading_frame_ws, si.normal_ws, si.wi_ws);
        }

        // Setup OpenPBR inputs.
        mi.inputs = openpbr_make_default_resolved_inputs();

        Attributes attributes = attributes_buffer.load<Attributes>(0);

        // Base
        mi.inputs.base_weight = eval_attribute(attributes.base_weight, si.uv, lod_sampler);
        mi.inputs.base_color = eval_attribute(attributes.base_color, si.uv, lod_sampler);
        mi.inputs.base_diffuse_roughness = eval_attribute(attributes.base_diffuse_roughness, si.uv, lod_sampler);
        mi.inputs.base_metalness = eval_attribute(attributes.base_metalness, si.uv, lod_sampler);

        // Specular
        mi.inputs.specular_weight = eval_attribute(attributes.specular_weight, si.uv, lod_sampler);
        mi.inputs.specular_color = eval_attribute(attributes.specular_color, si.uv, lod_sampler);
        mi.inputs.specular_roughness = eval_attribute(attributes.specular_roughness, si.uv, lod_sampler);
        mi.inputs.specular_roughness_anisotropy
            = eval_attribute(attributes.specular_roughness_anisotropy, si.uv, lod_sampler);
        mi.inputs.specular_ior = eval_attribute(attributes.specular_ior, si.uv, lod_sampler);

        // Coat
        mi.inputs.coat_weight = eval_attribute(attributes.coat_weight, si.uv, lod_sampler);
        mi.inputs.coat_color = eval_attribute(attributes.coat_color, si.uv, lod_sampler);
        mi.inputs.coat_roughness = eval_attribute(attributes.coat_roughness, si.uv, lod_sampler);
        mi.inputs.coat_roughness_anisotropy = eval_attribute(attributes.coat_roughness_anisotropy, si.uv, lod_sampler);
        mi.inputs.coat_ior = eval_attribute(attributes.coat_ior, si.uv, lod_sampler);
        mi.inputs.coat_darkening = eval_attribute(attributes.coat_darkening, si.uv, lod_sampler);

        // Fuzz
        mi.inputs.fuzz_weight = eval_attribute(attributes.fuzz_weight, si.uv, lod_sampler);
        mi.inputs.fuzz_color = eval_attribute(attributes.fuzz_color, si.uv, lod_sampler);
        mi.inputs.fuzz_roughness = eval_attribute(attributes.fuzz_roughness, si.uv, lod_sampler);

        // Transmission / volume
        mi.inputs.transmission_weight = eval_attribute(attributes.transmission_weight, si.uv, lod_sampler);
        mi.inputs.transmission_color = eval_attribute(attributes.transmission_color, si.uv, lod_sampler);
        mi.inputs.transmission_depth = eval_attribute(attributes.transmission_depth, si.uv, lod_sampler);
        mi.inputs.transmission_scatter = eval_attribute(attributes.transmission_scatter, si.uv, lod_sampler);
        mi.inputs.transmission_scatter_anisotropy
            = eval_attribute(attributes.transmission_scatter_anisotropy, si.uv, lod_sampler);
        mi.inputs.transmission_dispersion_scale
            = eval_attribute(attributes.transmission_dispersion_scale, si.uv, lod_sampler);
        mi.inputs.transmission_dispersion_abbe_number
            = eval_attribute(attributes.transmission_dispersion_abbe_number, si.uv, lod_sampler);

        // Thin-film
        mi.inputs.thin_film_weight = eval_attribute(attributes.thin_film_weight, si.uv, lod_sampler);
        mi.inputs.thin_film_thickness = eval_attribute(attributes.thin_film_thickness, si.uv, lod_sampler);
        mi.inputs.thin_film_ior = eval_attribute(attributes.thin_film_ior, si.uv, lod_sampler);

        // Emission
        mi.inputs.emission_luminance = eval_attribute(attributes.emission_luminance, si.uv, lod_sampler);
        mi.inputs.emission_color = eval_attribute(attributes.emission_color, si.uv, lod_sampler);

        // Geometry
        // inputs.geometry_opacity = 1.0f;
        mi.inputs.geometry_thin_walled = (header.flags & MaterialFlags::thin_walled) != MaterialFlags::none;

        // Geometry basis
        mi.inputs.geometry_basis.n = shading_frame_ws.normal;
        mi.inputs.geometry_basis.t = shading_frame_ws.tangent;
        mi.inputs.geometry_basis.b = shading_frame_ws.bitangent;
        mi.inputs.geometry_coat_basis = mi.inputs.geometry_basis;

        // Precompute emissive radiance for material instance
        mi.emissive_radiance = mi.inputs.emission_luminance * mi.inputs.emission_color;

        return mi;
    }

    public override bool is_emissive() { return has_emission; }

    public override MaterialFlags flags() { return header.flags; }

    public override uint2 compute_emission_sampling_grid(float2 uv[3]) { return uint2(100, 100); }

    public override float3 eval_emission<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        Attributes attributes = attributes_buffer.load<Attributes>(0);
        float emission_luminance = eval_attribute(attributes.emission_luminance, si.uv, lod_sampler);
        float3 emission_color = eval_attribute(attributes.emission_color, si.uv, lod_sampler);
        return emission_luminance * emission_color * float.PI;
    }

    public override float3 eval_emissive_radiance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        Attributes attributes = attributes_buffer.load<Attributes>(0);
        float emission_luminance = eval_attribute(attributes.emission_luminance, si.uv, lod_sampler);
        float3 emission_color = eval_attribute(attributes.emission_color, si.uv, lod_sampler);
        return emission_luminance * emission_color;
    }

    private float
    eval_attribute<LodSampler : ILodSampler>(const FloatAttribute attribute, float2 uv, const LodSampler lod_sampler)
    {
        return float(attribute.value)
            * attribute.texture_handle.sample(lod_sampler, uv, float(1.0))[attribute.texture_channel];
    }

    private float3
    eval_attribute<LodSampler : ILodSampler>(const Float3Attribute attribute, float2 uv, const LodSampler lod_sampler)
    {
        return float3(attribute.value) * attribute.texture_handle.sample(lod_sampler, uv, float4(1.0)).rgb;
    }
};
