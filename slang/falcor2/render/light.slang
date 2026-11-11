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
    /// Full probability density w.r.t. solid angle at the shading point, including emitter selection.
    public no_diff float pdf = {};
    /// Incident-radiance sampling weight (Li / pdf).
    public float3 weight = {};
    /// Capabilities and sampling-strategy properties of this sample.
    public no_diff LightSampleFlags flags = LightSampleFlags::none;
    /// Emitter type the sample is associated with.
    public no_diff EmitterType emitter_type = EmitterType::unknown;
    /// Emitter-specific identifier. Currently used for emissive triangle IDs.
    public no_diff uint emitter_id = uint(-1);
    /// Sampled finite emitter position in world-space, when applicable.
    public no_diff float3 pos_ws = {};
    /// Geometric normal at the sampled emitter position in world-space, when applicable.
    public no_diff float3 normal_ws = {};

    /// Returns true if any of the specified flags are set.
    public bool has_flag(const LightSampleFlags flags) { return is_set(this.flags, flags); }
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
    /// @param sg Sample generator.
    /// @param[out] sample The resulting light sample, including incident radiance divided by the PDF.
    /// @return True if a valid sample was generated, false otherwise.
    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    );

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
