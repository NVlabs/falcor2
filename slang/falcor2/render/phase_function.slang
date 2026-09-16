// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;

/// Result of sampling a phase function.
public struct PhaseFunctionSample
{
    /// Normalized outgoing direction in world space, pointing away from the scattering point.
    public float3 wo_ws = {};
    /// Probability density with respect to solid angle for sampling wo_ws.
    public float pdf = {};
    /// Spectral sample weight eval(wi_ws, wo_ws) / pdf.
    public float3 weight = {};
}

/// Interface for participating-media scattering phase functions.
///
/// Conventions:
/// - All directions are normalized and expressed in world space.
/// - Both wi_ws and wo_ws point away from the scattering point. wi_ws points toward the previous path vertex and
///   wo_ws points toward the next path vertex. Forward scattering therefore occurs near wo_ws = -wi_ws.
/// - Phase-function values and PDFs are densities with respect to solid angle.
/// - eval() may be spectral, while eval_pdf() is the scalar density used by sample().
public interface IPhaseFunction
{
    /// Evaluates the phase function.
    /// @param wi_ws Incident direction, pointing away from the scattering point.
    /// @param wo_ws Outgoing direction, pointing away from the scattering point.
    /// @return Spectral phase-function value with respect to solid angle.
    public float3 eval(const float3 wi_ws, const float3 wo_ws);

    /// Samples an outgoing direction from the phase function.
    /// @param wi_ws Incident direction, pointing away from the scattering point.
    /// @param[inout] sg Sample generator.
    /// @param[out] sample Phase-function sample. Only valid if the function returns true.
    /// @return Returns true if successful.
    public bool sample<S : ISampleGenerator>(const float3 wi_ws, inout S sg, out PhaseFunctionSample sample);

    /// Evaluates the probability density used to sample an outgoing direction.
    /// @param wi_ws Incident direction, pointing away from the scattering point.
    /// @param wo_ws Outgoing direction, pointing away from the scattering point.
    /// @return Probability density with respect to solid angle for sampling wo_ws.
    public float eval_pdf(const float3 wi_ws, const float3 wo_ws);
}
