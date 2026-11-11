// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;


/// Isotropic phase function representing uniform scattering over the sphere.
/// Its value and sampling density are constant at 1 / (4 * pi), independent of direction.
public struct IsotropicPhaseFunction : IPhaseFunction
{
    public __init() { }

    public float3 eval(const float3 wi, const float3 wo) { return float.INV_FOUR_PI; }

    public bool sample(const float3 wi, const PhaseFunctionSampleInput input, out PhaseFunctionSample sample)
    {
        sample.wo_ws = square_to_uniform_sphere(input.u_dir);
        sample.pdf = float.INV_FOUR_PI;
        sample.weight = 1.0;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo) { return float.INV_FOUR_PI; }
};
