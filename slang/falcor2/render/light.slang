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
    /// True if the emitter can only contribute through next-event estimation.
    public no_diff bool nee_only = {};
    /// Emitter type the sample is associated with.
    public no_diff EmitterType emitter_type = EmitterType::unknown;
    /// Emitter-specific identifier. Currently used for emissive triangle IDs.
    public no_diff uint emitter_id = uint(-1);
    /// Sampled finite emitter position in world-space, when applicable.
    public no_diff float3 pos_ws = {};
    /// Geometric normal at the sampled emitter position in world-space, when applicable.
    public no_diff float3 normal_ws = {};
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
};
