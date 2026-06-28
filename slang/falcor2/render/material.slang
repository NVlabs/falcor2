// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;

/// This is actually a sample from the MaterialInstance and would be more fittingly called
/// MaterialSample, but that name heavily clashes with the NeuralAppearance MaterialSample name
/// and we cannot currently use namespaces to really resolve the name collision, because the
/// namespace difference does not propagate to TypeReflection on the CPU.
public struct BSDFSample
{
    /// Sampled direction in world-space (normalized).
    public float3 wo_ws = {};
    /// pdf with respect to solid angle for the sampled direction (wo).
    public float pdf = {};
    /// Sample weight eval(wi, wo) / pdf(wo). Surface BSDF eval includes outgoing foreshortening.
    public float3 weight = {};
    /// Sampled lobe types.
    public LobeTypes lobe_types = LobeTypes::none;

    /// Return true if sample contains the specified lobe_type.
    public bool has_lobe(const LobeTypes lobe_type) { return is_set(lobe_types, lobe_type); }
}

public enum MaterialInstanceHints : uint {
    none = 0,
    /// Turn off normal mapping if supported.
    disable_normal_mapping = 0x1,
    /// Adjust shading normal to reduce risk of back-facing hits.
    adjust_shading_normal = 0x2,
};

public interface IMaterialInstance
{
    public float3 eval<S : ISampleGenerator>(const SurfaceInteraction si, const float3 wo_ws, inout S sg);

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out BSDFSample result);

    public float eval_pdf(const SurfaceInteraction si, const float3 wo_ws);

    /// Returns true if the material is emissive and thus can contribute to lighting.
    public bool is_emissive() { return false; }

    /// Evaluate emission (emissive radiance) towards the incident direction.
    /// @return Emissive radiance.
    public float3 eval_emissive_radiance() { return float3(0.0); }

    /// Returns true if the material represents an infinitely thin surface.
    public bool is_thin_walled() { return false; }

    public MaterialProperties collect_properties(const SurfaceInteraction si);

    /// Evaluate a non-physical, direction-independent material color guide.
    /// The default falls back to diffuse reflectance at normal incidence.
    public float3 eval_material_color(const SurfaceInteraction si)
    {
        SurfaceInteraction normal_si = si;
        normal_si.wi_ws = si.front_normal_ws();
        return saturate(collect_properties(normal_si).diffuse_reflection_albedo);
    }

    public Optional<ExtraBsdfProperties> collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        return {};
    }

    public LobeTypes get_lobe_types(const SurfaceInteraction si);

    // public bool has_volume_properties() { return false; }
    // public VolumeProperties volume_properties() { return {}; }
};

public interface IMaterial
{
    associatedtype MaterialInstance : IMaterialInstance;

    /// Setup a material instance for a given surface interaction.
    /// @param si Surface interaction.
    /// @param lod_sampler LOD sampler for sampling the emissive texture.
    /// @param hints Material instance hints (optional).
    /// @return A new material instance.
    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    );

    /// Returns true if the material is emissive and thus can contribute to lighting.
    public bool is_emissive() { return false; }

    /// Returns static material flags.
    public MaterialFlags flags() { return MaterialFlags::none; }

    /// Computes the resolution of the sampling grid for a triangle with the given UV coordinates.
    /// This is used to integrate emission across a triangle.
    /// @param uv UV coordinates of the triangle vertices.
    /// @return Sampling grid resolution (width, height).
    public uint2 compute_emission_sampling_grid(float2 uv[3]) { return uint2(1); }

    /// Evaluate emission (radiant exitance) for a given surface interaction.
    /// This is used to integrate emission across a surface.
    /// @param si Surface interaction.
    /// @param lod_sampler LOD sampler for sampling the emissive texture.
    /// @param hints Material instance hints.
    /// @return Emission (radiant exitance).
    public float3 eval_emission<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        return float3(0.0);
    }

    /// Evaluate emission (emissive radiance) for a given surface interaction.
    /// @param si Surface interaction.
    /// @param lod_sampler LOD sampler for sampling the emissive texture.
    /// @param hints Material instance hints.
    /// @return Emissive radiance.
    public float3 eval_emissive_radiance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        return float3(0.0);
    }

    // public VolumeProperties get_homogeneous_volume_properties(const uint MaterialType)
    // {
    //     return VolumeProperties(float3(0.f), float3(0.f), NullPhaseFunction());
    // }
};


// TODO(@tdavidovic): Added the types for training, to make sure that MaterialX instances can handle them.

public struct MaterialProperties
{
    public enum Flags : uint {
        none = 0x0,
        is_transmissive = 0x1,
        is_curve_scattering = 0x2,
        is_thin_walled = 0x4,
    };

    /// Radiance emitted in the incident direction (wi).
    public float3 emission = {};
    /// Surface roughness on a perceptually linear scale, where 0.0 = perfectly smooth and 1.0 =
    /// maximum roughness.
    public float roughness = {};
    /// Representative shading normal in world space that includes effects of normal mapping. Useful
    /// as guide for denoisers.
    public float3 guide_normal = {};
    ///< Flags storing additional properties.
    public Flags flags = {};

    // Approximate directional-hemispherical reflectance/transmittance of the BSDF (black-sky
    // albedo). The exact values are given by the integrals of the BSDF over wo given an incident
    // direction wi. The terms are separated into diffuse and non-diffuse components. Due to energy
    // conservation, the sum is expected to be <= 1.0.

    /// Directional-hemispherical diffuse reflectance. This is the ratio of total energy diffusely
    /// reflected to the energy incident along wi.
    public float3 diffuse_reflection_albedo = {};
    /// Directional-hemispherical diffuse transmittance. This is the ratio of total energy diffusely
    /// transmitted to the energy incident along wi.
    public float3 diffuse_transmission_albedo = {};
    /// Directional-hemispherical specular reflectance. This is the ratio of total energy
    /// non-diffusely reflected to the energy incident along wi.
    public float3 specular_reflection_albedo = {};
    /// Directional-hemispherical specular transmittance. This is the ratio of total energy
    /// non-diffusely transmitted to the energy incident along wi.
    public float3 specular_transmission_albedo = {};

    // Approximate specular reflectance. This is the color of specular reflection.
    // The diffuse reflectance is approximated as a Lambertian given by the diffuse albedo above.

    /// Specular reflectance at normal incidence (F0). This is in the range [0,1].
    public float3 specular_reflectance = {};

    public property is_transmissive : bool
    {
        get
        {
            return (flags & Flags::is_transmissive) != 0;
        }
    }

    public property is_curve_scattering : bool
    {
        get
        {
            return (flags & Flags::is_curve_scattering) != 0;
        }
    }

    public property is_thin_walled : bool
    {
        get
        {
            return (flags & Flags::is_thin_walled) != 0;
        }
    }
}

public struct ExtraBsdfProperties
{
    public __init() { }

    public float3 bsdf_N[16] = {};         ///< Normal used by each BSDF
    public float3 bsdf_T[16] = {};         ///< Tangent used by each BSDF
    public float3 bsdf_B[16] = {};         ///< Bitangent used by each BSDF
    public float3 bsdf_albedo[16] = {};    ///< Albedo of individual BSDFs
    public float3 bsdf_weight[16] = {};    ///< Weight for the given BSDF
    public float2 bsdf_roughness[16] = {}; ///< Anisotropic roughness
    public float3 bsdf_scratch[16] = {};   ///< Scratch direction and depth
    public float bsdf_index[16] = {};      ///< A BSDF index, e.g. to differentiate between different MERL materials.

    public uint bsdf_count = 0; ///< Number of actually active BSDFs (0 == invalid)
}

public struct VolumeProperties
{
    public float3 sigma_a = {};           ///< Absorption coefficient.. or maybe cross section.
    public float3 sigma_s = {};           ///< Scattering coefficient.. or maybe cross section.
    public IPhaseFunction phase_function; ///< Set to NullPhaseFunction when invalid

    public __init() { phase_function = NullPhaseFunction(); }
    public __init(float3 sigma_a_, float3 sigma_s_, IPhaseFunction phase_function_)
    {
        this.sigma_a = sigma_a_;
        this.sigma_s = sigma_s_;
        this.phase_function = phase_function_;
    }
}
