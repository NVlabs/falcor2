// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;

/// Shares the uniform sample coordinates and auxiliary random bits documented by BSDFSampleInput.
/// No coordinate-frame conversion is needed when forwarding these inputs to a BSDF.
public typealias MaterialSampleInput = BSDFSampleInput;

/// Consume exactly four scalar dimensions using the BSDF input field order.
public MaterialSampleInput make_material_sample_input<S : ISampleGenerator>(inout S sg)
{
    return make_bsdf_sample_input(sg);
}

/// Result of a material sample. Contains the sampled direction, pdf, weight, and BSDF flags.
public struct MaterialSample
{
    /// Sampled direction in world-space (normalized).
    public float3 wo_ws = {};
    /// Marginal pdf with respect to solid angle for the sampled direction (wo), zero for delta events.
    public float pdf = {};
    /// Continuation estimator weight. Under eval_consistent, continuous weights equal the selected component value
    /// divided by its joint component/direction PDF. Native weights need not satisfy that identity.
    /// The pdf field remains marginal. Delta events carry integrated weights with a zero-PDF sentinel.
    public float3 weight = {};
    /// Flags describing the sampled scattering event.
    public BSDFFlags flags = BSDFFlags::none;
    /// Contribution selector; eval_consistent guarantees that weight reconstructs this contribution.
    /// Native estimators may return full_response without satisfying the reconstruction identity.
    /// Other indices are meaningful only within a compatible material component layout.
    public BSDFComponent component = BSDFComponent::full_response;

    /// Return true if the sample contains any of the specified flags.
    public bool has_flag(const BSDFFlags flags) { return is_set(this.flags, flags); }
}

public enum class MaterialInstanceHints : uint {
    none = 0,
    /// Turn off normal mapping if supported.
    disable_normal_mapping = 0x1,
    /// Adjust shading normal to reduce risk of back-facing hits.
    adjust_shading_normal = 0x2,
};

/// Renderer requirement for material continuation weights; does not change the physical scattering function.
public enum class MaterialSampleWeightPolicy : uint {
    /// Preserve native estimators, which need not reconstruct through component evaluation.
    native = 0,
    /// Require continuous weights to equal component value divided by joint component/direction density.
    /// Full-response adapters may need an additional evaluation. Delta weights retain their integrated convention.
    eval_consistent = 1,
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

/// Minimal writer contract used by materials to publish numeric AOVs.
public interface IMaterialAOVWriter
{
    public static void generic_data(uint2 coord, uint id, float4 value);
    public static bool generic_data_enabled(uint id);
}

/// Scattering and emission at a geometrically accepted surface hit.
public interface IMaterialInstance
{
    /// Evaluates the selected scattering contribution. Convenience alias for eval_with_pdf().
    /// @param si Accepted surface interaction, including the incident direction in world space.
    /// @param wo_ws Normalized outgoing direction in world space, pointing away from the surface.
    /// @param component Contribution to evaluate, or full_response for the entire response.
    /// @return Selected scattering value using the surface or curve BSDF convention.
    public float3
    eval(const SurfaceInteraction si, const float3 wo_ws, const BSDFComponent component = BSDFComponent::full_response)
    {
        return eval_with_pdf<BSDFEvalHints::eval>(si, wo_ws, component).value;
    }

    /// Evaluates value and density together, sharing work when both are requested.
    /// Component values include all material weights; their joint PDFs include selection probabilities.
    /// Full response uses the marginal directional PDF. Full-response-only materials ignore the selector;
    /// component-aware materials return zero for unknown indices.
    /// @tparam Hints Compile-time selection of the value and PDF fields to compute.
    /// @param si Accepted surface interaction, including the incident direction in world space.
    /// @param wo_ws Normalized outgoing direction in world space, pointing away from the surface.
    /// @param component Contribution to evaluate, or full_response for the entire response and marginal PDF.
    /// @return Requested scattering value and solid-angle density; unrequested fields must not be used.
    public BSDFEval eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both>(
        const SurfaceInteraction si,
        const float3 wo_ws,
        const BSDFComponent component = BSDFComponent::full_response
    );

    /// Evaluates the selected joint density, or the marginal density for full response.
    /// @param si Accepted surface interaction, including the incident direction in world space.
    /// @param wo_ws Normalized outgoing direction in world space, pointing away from the surface.
    /// @param component Contribution whose joint density is requested, or full_response for the marginal density.
    /// @return Solid-angle density including component selection probability, or zero for delta events.
    public float eval_pdf(
        const SurfaceInteraction si,
        const float3 wo_ws,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        return eval_with_pdf<BSDFEvalHints::pdf>(si, wo_ws, component).pdf;
    }

    /// Equal nonzero IDs guarantee identical component-index meanings across material instances.
    /// Zero is reserved for materials whose samples always represent full response. IDs are scene-local.
    /// @return Component layout ID for compatibility checks during reuse.
    public uint get_component_layout_id() { return 0; }

    /// Reconnection heuristic using a component at the current incident direction and its marginal PDF.
    /// The default uses density as a proxy; full-response groups follow the same policy.
    /// Callers must exclude delta events using the sample flags before applying this heuristic.
    /// @param si Accepted surface interaction at which the contribution is classified.
    /// @param component Contribution to classify, or full_response for the entire response.
    /// @param marginal_pdf Marginal directional sampling density at the current interaction and outgoing direction.
    /// @param roughness_threshold Roughness cutoff used by the reconnection heuristic.
    /// @return True if the contribution is classified as rough enough for reconnection.
    public bool is_component_rough(
        const SurfaceInteraction si,
        BSDFComponent component,
        float marginal_pdf,
        float roughness_threshold
    )
    {
        return roughness_threshold <= 0.0 || marginal_pdf <= 1.0 / (roughness_threshold * roughness_threshold);
    }

    /// Sample scattering using caller-owned coordinates and auxiliary seed bits.
    /// Returns false on rejection; sample is valid only on success.
    /// MATERIAL_SAMPLE_WEIGHT_POLICY selects whether weights must reconstruct through component evaluation.
    /// @param si Accepted surface interaction, including the incident direction in world space.
    /// @param input Uniform sample coordinates and auxiliary random bits; see MaterialSampleInput.
    /// @param[out] sample Sampled direction, weight, marginal PDF, event flags and contribution identity.
    /// @return True if a valid sample was generated.
    public bool sample(const SurfaceInteraction si, const MaterialSampleInput input, out MaterialSample sample);

    /// Returns true if the material is emissive and thus can contribute to lighting.
    public bool is_emissive() { return false; }

    /// Evaluate emission (emissive radiance) towards the incident direction.
    /// @return Emissive radiance.
    public float3 eval_emissive_radiance() { return float3(0.0); }

    /// Return the absolute IOR of the homogeneous medium bounded by this material.
    public float get_interior_ior() { return 1.0; }

    /// Return the resolved homogeneous medium bounded by this material.
    public HomogeneousMediumProperties get_interior_medium() { return {}; }

    /// Collect material properties used as rendering and denoising guides.
    /// @param si Accepted surface interaction at which properties are evaluated.
    /// @return Representative material properties; these may approximate the full scattering model.
    public MaterialProperties collect_properties(const SurfaceInteraction si);

    /// Collect optional per-BSDF diagnostic properties.
    /// @param si Accepted surface interaction at which properties are evaluated.
    /// @param wo Outgoing direction in world space.
    /// @return Per-BSDF properties when supported, otherwise an empty optional.
    public Optional<ExtraBsdfProperties> collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        return {};
    }

    /// Query the scattering event types available at an interaction.
    /// @param si Accepted surface interaction at which event types are evaluated.
    /// @return Flags describing the available scattering event types.
    public BSDFFlags get_lobe_types(const SurfaceInteraction si);
};

public interface IMaterial
{
    associatedtype MaterialInstance : IMaterialInstance;

    /// Evaluate and write material-authored numeric outputs when the compiled capability is enabled,
    /// without constructing a scattering instance.
    public void write_material_aovs<Writer : IMaterialAOVWriter, LodSampler : ILodSampler>(
        uint2 coord,
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        if (ENABLE_MATERIAL_AOVS)
            write_material_aovs_impl<Writer, LodSampler>(coord, si, lod_sampler, hints);
    }

    /// Producer hook; consumers call write_material_aovs() to respect the compiled capability.
    public void write_material_aovs_impl<Writer : IMaterialAOVWriter, LodSampler : ILodSampler>(
        uint2 coord,
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
    }

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
