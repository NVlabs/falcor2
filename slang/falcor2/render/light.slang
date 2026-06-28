// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Interface for scene light sources.
/// Implementations represent specific light types (e.g., point, directional, environment)
/// and provide methods for evaluating radiance, sampling directions, and computing PDFs.
public interface ILight
{
    /// Evaluates the incident radiance arriving at the interaction point from the sampled direction.
    /// @param ref The interaction (shading point) being illuminated.
    /// @param sample The direction sample describing the light direction and hit point.
    /// @return The incident radiance from the light along the sampled direction.
    public float3 eval_direction<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample);

    /// Samples a direction toward the light source from the given interaction point.
    /// @param ref The interaction (shading point) from which to sample.
    /// @param sg Sample generator.
    /// @param[out] sample The resulting direction sample describing the sampled point on the light.
    /// @param[out] weight The sample weight (incident radiance divided by the PDF).
    /// @return True if a valid sample was generated, false otherwise.
    public bool sample_direction<Interaction : IInteraction, SampleGenerator : ISampleGenerator>(
        const Interaction ref,
        inout SampleGenerator sg,
        no_diff out DirectionSample sample,
        out float3 weight
    );

    /// Evaluates the probability density function for a given direction sample.
    /// @param ref The interaction (shading point) from which the sample was taken.
    /// @param sample The direction sample to evaluate.
    /// @return The PDF value with respect to solid angle at the shading point.
    public float eval_direction_pdf<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample);
};
