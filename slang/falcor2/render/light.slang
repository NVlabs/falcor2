// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Emitter types a light sample can be associated with.
public enum class EmitterType : uint {
    unknown,
    analytic,
    environment,
    emissive_triangle,
};

/// Capabilities and sampling-strategy properties of a light sample.
public enum class LightSampleFlags : uint {
    none = 0,
    /// pos_ws and normal_ws identify a sampled surface endpoint.
    has_surface_point = 1 << 0,
    /// The emitter can only contribute through next-event estimation.
    nee_only = 1 << 1,
};

/// Context for sampling incident illumination at a shading point.
public struct LightSampleContext
{
    /// Shading point position in world-space.
    public float3 pos_ws = {};
    /// Receiver normal defining the hemisphere to importance sample, or zero when sampling is unrestricted.
    public no_diff float3 normal_ws = {};
    public __init() { }

    public __init(const SurfaceInteraction interaction)
    {
        pos_ws = interaction.position_ws;
        normal_ws = interaction.normal_ws;
    }
};

/// Result of sampling incident illumination from a scene light.
public struct LightSample
{
    /// Sampled direction from the shading point toward the light in world-space (normalized).
    public no_diff float3 dir_ws = {};
    /// Distance used for the visibility ray.
    public no_diff float distance = {};
    /// Probability density w.r.t. solid angle, or discrete probability for point lights.
    /// Includes emitter selection for ILightSampler results; ILight results are conditional on the light.
    public no_diff float pdf = {};
    /// Unoccluded incident radiance, retained even when the current proposal has zero density.
    public float3 radiance = {};
    /// Incident-radiance sampling weight (Li / pdf), or zero when pdf is zero.
    public float3 weight = {};
    /// Capabilities and sampling-strategy properties of this sample.
    public no_diff LightSampleFlags flags = LightSampleFlags::none;
    /// Emitter type the sample is associated with.
    public no_diff EmitterType emitter_type = EmitterType::unknown;
    /// Scene light or emissive triangle ID; individual ILight samples leave this invalid.
    public no_diff uint emitter_id = uint(-1);
    /// Sampled finite emitter position in world-space, when applicable.
    public no_diff float3 pos_ws = {};
    /// Geometric normal at the sampled emitter position in world-space, when applicable.
    public no_diff float3 normal_ws = {};

    /// Returns true if any of the specified flags are set.
    public bool has_flag(const LightSampleFlags flags) { return is_set(this.flags, flags); }
};

/// Retained emitter endpoint, independent of the receiver and of the sampling distribution.
/// Valid only while scene emitter identities and parameterizations remain unchanged. No visibility or PDF is stored.
/// Individual lights leave emitter_id invalid; scene samplers assign it. Environment references store world direction
/// and scene-level evaluation uses the entire environment mixture, independent of the selected component.
/// Evaluation trusts the emitter identity and parameters; callers must supply a valid reference.
public struct LightSampleRef
{
    public EmitterType emitter_type = EmitterType::unknown;
    public uint emitter_id = uint(-1);
    /// Triangle barycentrics.yz, disk/rectangle local coordinates, or cylinder angle/height fractions.
    /// Sphere world-space surface normals and infinite world-space directions use ndir_to_oct_unorm() encoding.
    /// Unused components are zero.
    public float2 parameters = {};

    /// Whether an endpoint was retained. Does not validate the emitter identity or parameters.
    public bool is_valid() { return emitter_type != EmitterType::unknown; }
};

/// Interface for scene light sources.
/// Implementations represent specific light types (e.g., point, directional, environment)
/// and provide methods for evaluating radiance, sampling directions, and computing PDFs.
public interface ILight
{
    /// Returns a non-negative luminance-weighted estimate used for light selection.
    /// RGB values are reduced by luminance(), which uses Rec.709 coefficients.
    /// This is a scalar reduction of renderer RGB radiometry, not a photometric value in lumens.
    /// For finite lights this is total emitted radiant flux (for example, W when authored in physical units).
    /// For infinite lights this is the integral of incident radiance over solid angle, with units of radiant flux
    /// per projected area. The two domains have different units and must not be compared directly.
    public float power_estimate();

    /// Evaluates environment radiance along a ray.
    /// Returns zero by default for lights that do not contribute at infinity.
    /// @param ctx The shading point context.
    /// @param dir_ws Direction from the shading point toward the environment.
    public float3 eval_environment_radiance(const LightSampleContext ctx, const float3 dir_ws) { return float3(0.0); }

    /// Samples incident illumination from the light source.
    /// @param ctx The shading point context.
    /// @param u Uniform sample in [0, 1)^2 for sampling a position or direction on the light.
    /// @param[out] sample The resulting light sample, including incident radiance divided by the PDF.
    /// @param[out] sample_ref Retained endpoint on success, except virtual sphere proxy samples return an invalid
    /// reference while preserving successful ordinary sampling. Outputs must not be retained after sampling fails.
    /// @return True if a valid sample was generated, false otherwise.
    public bool
    sample(const LightSampleContext ctx, const float2 u, out LightSample sample, out LightSampleRef sample_ref);

    /// Compatibility overload that discards the retained reference.
    public bool sample(const LightSampleContext ctx, const float2 u, out LightSample sample)
    {
        LightSampleRef sample_ref;
        return this.sample(ctx, u, sample, sample_ref);
    }

    /// Reconstructs an endpoint without consuming random numbers. Requires a valid reference belonging to this light.
    /// Returns false for degenerate connections or unsupported receiver contexts, not for zero radiance or PDF.
    /// Computes the current conditional proposal density, not the density at the original sampling context.
    /// Receiver-dependent virtual sphere proxies are unsupported, both as source and destination.
    public bool eval_sample(const LightSampleContext ctx, const LightSampleRef sample_ref, out LightSample sample);

    /// Evaluates the PDF for sampling a direction from the shading point.
    /// Returns zero by default for lights without a complementary direction-sampling strategy.
    /// @param ctx The shading point context.
    /// @param dir_ws Direction from the shading point toward the light.
    /// @return The PDF value with respect to solid angle at the shading point.
    public float eval_pdf(const LightSampleContext ctx, const float3 dir_ws) { return 0.0; }

    /// Intersects the authored finite light surface when it corresponds to the surface sampled from the ray origin.
    /// Returns false for lights without such a surface or when sampling uses a distinct NEE-only proxy.
    public bool intersect_surface(const Ray ray, out float t)
    {
        t = {};
        return false;
    }

    /// Returns the outward geometric normal at a known authored finite light-surface point.
    /// Returns zero for lights without a finite surface.
    public float3 eval_surface_normal(const float3 pos_ws) { return {}; }

    /// Evaluates emitted radiance from a known authored finite light-surface point toward a receiver.
    /// Returns zero for lights without a finite surface or when the receiver is behind a one-sided emitter.
    public float3
    eval_surface_radiance(const float3 receiver_pos_ws, const float3 emitter_pos_ws, const float3 emitter_normal_ws)
    {
        return {};
    }

    /// Evaluates the conditional solid-angle PDF induced by the light's surface sampling strategy at a known
    /// authored finite light-surface point. Returns zero when the point has no density under the surface sampler for
    /// this receiver context.
    /// The PDF excludes any scene-level light-selection probability.
    public float
    eval_surface_sampling_pdf(const LightSampleContext ctx, const float3 emitter_pos_ws, const float3 emitter_normal_ws)
    {
        return 0.0;
    }
};

/// Evaluates an infinite-light reference in world-space without replaying its sampling warp.
public bool eval_environment_light_sample<Light : ILight>(
    Light light,
    const LightSampleContext ctx,
    const LightSampleRef sample_ref,
    out LightSample sample
)
{
    sample = {};
    sample.dir_ws = oct_to_ndir_unorm(sample_ref.parameters);
    sample.distance = float.MAX;
    sample.emitter_type = sample_ref.emitter_type;
    sample.emitter_id = sample_ref.emitter_id;
    sample.radiance = light.eval_environment_radiance(ctx, sample.dir_ws);
    sample.pdf = light.eval_pdf(ctx, sample.dir_ws);
    sample.weight = sample.pdf > 0.0 ? sample.radiance / sample.pdf : float3(0.0);
    return true;
}
