// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;

/// Result of a material sample. Contains the sampled direction, pdf, weight, and BSDF flags.
public struct MaterialSample
{
    /// Sampled direction in world-space (normalized).
    public float3 wo_ws = {};
    /// pdf with respect to solid angle for the sampled direction (wo).
    public float pdf = {};
    /// Sample weight eval(wi, wo) / pdf(wo). Surface BSDF eval includes outgoing foreshortening.
    public float3 weight = {};
    /// Flags describing the sampled scattering event.
    public BSDFFlags flags = BSDFFlags::none;

    /// Return true if the sample contains any of the specified flags.
    public bool has_flag(const BSDFFlags flags) { return is_set(this.flags, flags); }
}

public enum MaterialInstanceHints : uint {
    none = 0,
    /// Turn off normal mapping if supported.
    disable_normal_mapping = 0x1,
    /// Adjust shading normal to reduce risk of back-facing hits.
    adjust_shading_normal = 0x2,
};

/// Resolved properties of a homogeneous participating medium.
/// Absorption and scattering coefficients are expressed in inverse scene units.
/// Zero coefficients represent vacuum.
public struct HomogeneousMediumProperties
{
    public float3 sigma_a = {};
    public float3 sigma_s = {};
    public float anisotropy = 0.0;
};

/// Scattering and emission at a geometrically accepted surface hit.
public interface IMaterialInstance
{
    /// Evaluates the scattering value. This is a convenience alias for eval_with_pdf().
    public float3 eval<S : ISampleGenerator>(const SurfaceInteraction si, const float3 wo_ws, inout S sg)
    {
        return eval_with_pdf<BSDFEvalHints::eval>(si, wo_ws, sg).value;
    }

    /// Evaluates the scattering value and directional PDF together.
    /// Prefer this operation when both results are needed, so shared work is performed only once.
    /// The sample generator may affect only the value calculation; the PDF must be deterministic and
    /// generator-independent.
    public BSDFEval eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both, S : ISampleGenerator>(
        const SurfaceInteraction si,
        const float3 wo_ws,
        inout S sg
    );

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out MaterialSample sample);

    /// Evaluates the directional PDF. This is a convenience alias for eval_with_pdf().
    public float eval_pdf(const SurfaceInteraction si, const float3 wo_ws)
    {
        EvalPdfFallbackSampleGenerator sg = EvalPdfFallbackSampleGenerator(0u);
        return eval_with_pdf<BSDFEvalHints::pdf>(si, wo_ws, sg).pdf;
    }

    /// Returns true if the material is emissive and thus can contribute to lighting.
    public bool is_emissive() { return false; }

    /// Evaluate emission (emissive radiance) towards the incident direction.
    /// @return Emissive radiance.
    public float3 eval_emissive_radiance() { return float3(0.0); }

    /// Return the absolute IOR of the homogeneous medium bounded by this material.
    public float get_interior_ior() { return 1.0; }

    /// Return the resolved homogeneous medium bounded by this material.
    public HomogeneousMediumProperties get_interior_medium() { return {}; }

    public MaterialProperties collect_properties(const SurfaceInteraction si);

    public Optional<ExtraBsdfProperties> collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        return {};
    }

    public BSDFFlags get_lobe_types(const SurfaceInteraction si);
};

public interface IMaterial
{
    associatedtype MaterialInstance : IMaterialInstance;

    /// Setup a material instance for a given surface interaction.
    /// @param si Surface interaction.
    /// @param lod_sampler LOD sampler used while constructing the instance.
    /// @param hints Material instance hints.
    /// @return A material instance conditional on an accepted surface hit.
    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    );

    /// Returns true if the material is emissive and thus can contribute to lighting.
    public bool is_emissive() { return false; }

    /// Returns static material flags.
    public MaterialFlags flags() { return MaterialFlags::none; }

    /// Evaluate cutout opacity for a surface interaction.
    /// This is called during intersection traversal when the material header selects
    /// material opacity evaluation instead of the packed factor and texture source.
    public float eval_opacity<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        return 1.0;
    }

    /// Computes the resolution of the sampling grid for a triangle with the given UV coordinates.
    /// This is used to integrate emission across a triangle.
    /// @param uv UV coordinates of the triangle vertices.
    /// @return Sampling grid resolution (width, height).
    public uint2 compute_emission_sampling_grid(float2 uv[3]) { return uint2(1); }

    /// Evaluate emission (radiant exitance) for a given surface interaction.
    /// For one-sided Lambertian emission, radiant exitance is pi times emitted radiance.
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
    /// Representative non-physical material color guide.
    public float3 material_color = {};
    /// Radiance emitted in the incident direction (wi).
    public float3 emission = {};
    /// Metallic workflow parameter, if the material has one.
    public float metallic = {};
    /// Surface roughness on a perceptually linear scale, where 0.0 = perfectly smooth and 1.0 =
    /// maximum roughness.
    public float roughness = {};
    /// Representative shading normal in world space that includes effects of normal mapping. Useful
    /// as guide for denoisers.
    public float3 guide_normal = {};

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
}

public struct ExtraBsdfProperties
{
    public static const int max_bsdf_count = 16;

    public __init() { }

    public float3 bsdf_N[max_bsdf_count] = {};         ///< Normal used by each BSDF
    public float3 bsdf_T[max_bsdf_count] = {};         ///< Tangent used by each BSDF
    public float3 bsdf_B[max_bsdf_count] = {};         ///< Bitangent used by each BSDF
    public float3 bsdf_albedo[max_bsdf_count] = {};    ///< Albedo of individual BSDFs
    public float3 bsdf_weight[max_bsdf_count] = {};    ///< Weight for the given BSDF
    public float2 bsdf_roughness[max_bsdf_count] = {}; ///< Anisotropic roughness
    public float3 bsdf_scratch[max_bsdf_count] = {};   ///< Scratch direction and depth
    public float bsdf_index[max_bsdf_count] = {};      ///< A BSDF index, e.g. to differentiate between different MERL
                                                       ///< materials.

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
