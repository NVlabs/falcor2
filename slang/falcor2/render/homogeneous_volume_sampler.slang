// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Scattering-rate distance sampler for homogeneous media.
///
/// Absorption is integrated into the event and surface weights instead of being
/// sampled as a discrete event. RGB scattering coefficients form a throughput-
/// weighted mixture proposal.
public struct HomogeneousVolumeSampler
{
    private float3 sigma_a;
    private float3 sigma_s;

    public __init(const HomogeneousMediumProperties medium)
    {
        sigma_a = medium.sigma_a;
        sigma_s = medium.sigma_s;
    }

    /// Return true when the medium can produce a scattering event.
    public bool has_scattering() { return any(sigma_s > 0.0); }

    /// Compute probabilities for selecting an RGB scattering-distance proposal.
    public float3 channel_probabilities(const float3 throughput)
    {
        const float3 sigma_t = sigma_a + sigma_s;
        const float3 albedo = float3(
            sigma_s.x > 0.0 ? sigma_s.x / sigma_t.x : 1.0,
            sigma_s.y > 0.0 ? sigma_s.y / sigma_t.y : 1.0,
            sigma_s.z > 0.0 ? sigma_s.z / sigma_t.z : 1.0
        );

        float3 probabilities = max(throughput, 0.0) * albedo;
        const float probability_sum = probabilities.x + probabilities.y + probabilities.z;
        if (!(probability_sum >= 1e-8))
            return float3(1.0 / 3.0);
        return probabilities / probability_sum;
    }

    /// Sample a scattering distance. Returns false only when the medium cannot scatter.
    /// A selected zero-scattering channel represents probability mass at infinity.
    public bool sample_distance<S : ISampleGenerator>(const float3 throughput, inout S sg, out float distance)
    {
        distance = float.INFINITY;
        if (!has_scattering())
            return false;

        const float3 probabilities = channel_probabilities(throughput);
        const float channel_sample = next_float(sg);
        const uint channel
            = channel_sample < probabilities.x ? 0 : (channel_sample < probabilities.x + probabilities.y ? 1 : 2);
        const float sigma_s = sigma_s[channel];
        if (sigma_s <= 0.0)
            return true;

        const float distance_sample = min(next_float(sg), 0.999999940f);
        distance = -log(1.0 - distance_sample) / sigma_s;
        return true;
    }

    /// Evaluate Beer-Lambert transmittance over a finite non-negative distance.
    public float3 eval_transmittance(const float distance)
    {
        const float3 sigma_t = sigma_a + sigma_s;
        return float3(
            sigma_t.x > 0.0 ? exp(-sigma_t.x * distance) : 1.0,
            sigma_t.y > 0.0 ? exp(-sigma_t.y * distance) : 1.0,
            sigma_t.z > 0.0 ? exp(-sigma_t.z * distance) : 1.0
        );
    }

    /// Evaluate transmittance to infinity without multiplying zero extinction by an infinite distance.
    public float3 eval_transmittance_to_infinity()
    {
        return select((sigma_a + sigma_s) > 0.0, float3(0.0), float3(1.0));
    }

    /// Evaluate the throughput weight for a sampled scattering event.
    public float3 eval_event_weight(const float3 throughput, const float distance)
    {
        const float3 probabilities = channel_probabilities(throughput);
        const float3 proposal_transmittance = exp(-sigma_s * distance);
        const float proposal_pdf = dot(probabilities, proposal_transmittance * sigma_s);
        if (!(proposal_pdf > 0.0))
            return float3(0.0);
        return eval_transmittance(distance) * sigma_s / proposal_pdf;
    }

    /// Evaluate the throughput weight when a surface precedes the sampled event.
    public float3 eval_surface_weight(const float3 throughput, const float distance)
    {
        const float3 probabilities = channel_probabilities(throughput);
        const float3 proposal_transmittance = exp(-sigma_s * distance);
        const float proposal_survival = dot(probabilities, proposal_transmittance);
        if (!(proposal_survival > 0.0))
            return float3(0.0);
        return eval_transmittance(distance) / proposal_survival;
    }

    /// Evaluate the throughput weight when the sampled distance escapes to infinity.
    public float3 eval_surface_weight_to_infinity(const float3 throughput)
    {
        const float3 probabilities = channel_probabilities(throughput);
        const float3 proposal_transmittance = select(sigma_s > 0.0, float3(0.0), float3(1.0));
        const float proposal_survival = dot(probabilities, proposal_transmittance);
        if (!(proposal_survival > 0.0))
            return float3(0.0);
        return eval_transmittance_to_infinity() / proposal_survival;
    }
}
