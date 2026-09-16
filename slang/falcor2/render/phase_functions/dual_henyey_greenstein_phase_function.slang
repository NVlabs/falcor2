// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Spectral mixture of two Henyey-Greenstein phase functions.
///
/// Each color channel evaluates w0 * HG(mean_cosine[0]) + (1 - w0) * HG(mean_cosine[1]). Sampling uses a scalar
/// mixture formed from the arithmetic channel averages of the weights and mean cosines. The returned spectral weight
/// accounts for the difference between that scalar sampling density and the spectral phase-function value.
public struct DualHenyeyGreensteinPhaseFunction : IPhaseFunction
{
    /// Spectral mean cosine of each Henyey-Greenstein lobe.
    float3 mean_cosine[2];
    /// Spectral weight of the first lobe. The second lobe has weight 1 - w0.
    float3 w0;

    /// Constructs a two-lobe Henyey-Greenstein mixture.
    /// @param mean_cosine_0 Spectral mean cosine of the first lobe, with components in (-1, 1).
    /// @param mean_cosine_1 Spectral mean cosine of the second lobe, with components in (-1, 1).
    /// @param w0 Spectral mixture weight of the first lobe, with components in [0, 1].
    public __init(float3 mean_cosine_0, float3 mean_cosine_1, float3 w0)
    {
        this.mean_cosine[0] = mean_cosine_0;
        this.mean_cosine[1] = mean_cosine_1;
        this.w0 = w0;
    }

    public float3 eval(const float3 wi, const float3 wo)
    {
        return eval_henyey_greenstein(dot(wi, wo), mean_cosine[0]) * w0
            + eval_henyey_greenstein(dot(wi, wo), mean_cosine[1]) * (1.0 - w0);
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, inout S sg, out PhaseFunctionSample sample)
    {
        const float3 u = next_float3(sg);

        const float average_w0 = (w0[0] + w0[1] + w0[2]) / 3.0;
        const float3 selected_mean_cosine = u.z <= average_w0 ? mean_cosine[0] : mean_cosine[1];
        // Use a scalar lobe distribution obtained by averaging its spectral mean cosine.
        const float average_mean_cosine
            = (selected_mean_cosine[0] + selected_mean_cosine[1] + selected_mean_cosine[2]) / 3.0;

        float cos_theta;
        if (abs(average_mean_cosine) < HENYEY_GREENSTEIN_ISOTROPIC_THRESHOLD) {
            cos_theta = 1.0 - 2.0 * u.x;
        } else {
            const float sqr = (1.0 - average_mean_cosine * average_mean_cosine)
                / (1.0 - average_mean_cosine + 2.0 * average_mean_cosine * u.x);
            cos_theta = (1.0 + average_mean_cosine * average_mean_cosine - sqr * sqr) / (2.0 * average_mean_cosine);
        }

        const float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        const float phi = float.TWO_PI * u.y;
        const float3 w = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

        const float3 dir = -wi;
        float3 t, b;
        branchless_ONB(dir, t, b);

        sample.wo_ws = normalize(w.x * t + w.y * b + w.z * dir);
        sample.pdf = eval_pdf(wi, sample.wo_ws);
        sample.weight = eval(wi, sample.wo_ws) / sample.pdf;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo)
    {
        const float mean_cosine_0 = (mean_cosine[0][0] + mean_cosine[0][1] + mean_cosine[0][2]) / 3.0;
        const float mean_cosine_1 = (mean_cosine[1][0] + mean_cosine[1][1] + mean_cosine[1][2]) / 3.0;
        const float average_w0 = (w0[0] + w0[1] + w0[2]) / 3.0;
        return eval_henyey_greenstein_sampling_pdf(dot(wi, wo), mean_cosine_0) * average_w0
            + eval_henyey_greenstein_sampling_pdf(dot(wi, wo), mean_cosine_1) * (1.0 - average_w0);
    }
}
