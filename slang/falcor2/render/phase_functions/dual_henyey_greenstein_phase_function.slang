// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct DualHenyeyGreensteinPhaseFunction : IPhaseFunction
{
    float3 meancosine[2];
    float3 w0;

    public __init(float3 meancosine0, float3 meancosine1, float3 w0)
    {
        this.meancosine[0] = meancosine0;
        this.meancosine[1] = meancosine1;
        this.w0 = w0;
    }

    public float3 eval(const float3 wi, const float3 wo)
    {
        return evalHenyeyGreenstein(dot(wi, wo), meancosine[0]) * w0
            + evalHenyeyGreenstein(dot(wi, wo), meancosine[1]) * (1.f - w0);
    }

    public bool
    sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, inout S sg)
    {
        const float3 u = next_float3(sg);

        const float avgW0 = (w0[0] + w0[1] + w0[2]) / 3.f;
        const float3 meancosineSelect = u.z <= avgW0 ? meancosine[0] : meancosine[1];
        // Taking the average is really not the smartest thing.
        const float meancosineSelectAvg = (meancosineSelect[0] + meancosineSelect[1] + meancosineSelect[2]) / 3.f;

        float cosTheta;
        if (abs(meancosineSelectAvg) < 1e-3f) {
            cosTheta = 1.f - 2.f * u.x;
        } else {
            const float sqr = (1.f - meancosineSelectAvg * meancosineSelectAvg)
                / (1.f - meancosineSelectAvg + 2.f * meancosineSelectAvg * u.x);
            cosTheta = (1.f + meancosineSelectAvg * meancosineSelectAvg - sqr * sqr) / (2.f * meancosineSelectAvg);
        }

        float selectionWeight = {};
        float3 pfWeight = {};
        if (u.z <= avgW0) {
            selectionWeight = avgW0;
            pfWeight = w0;
        } else {
            selectionWeight = (1.0 - avgW0);
            pfWeight = 1.f - w0;
        }

        const float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        const float phi = float.TWO_PI * u.y;
        const float3 w = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

        const float3 dir = -wi;
        float3 T, B;
        branchless_ONB(dir, T, B);

        wo = normalize(w.x * T + w.y * B + w.z * dir);
        pdf = evalHenyeyGreenstein(-cosTheta, meancosineSelectAvg);
        weight = (evalHenyeyGreenstein(-cosTheta, meancosineSelect) * pfWeight
                  / evalHenyeyGreenstein(-cosTheta, meancosineSelectAvg))
            / selectionWeight;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo)
    {
        const float mc0 = (meancosine[0][0] + meancosine[0][1] + meancosine[0][2]) / 3.f;
        const float mc1 = (meancosine[1][0] + meancosine[1][1] + meancosine[1][2]) / 3.f;
        const float avgW0 = (w0[0] + w0[1] + w0[2]) / 3.f;
        return evalHenyeyGreenstein(dot(wi, wo), mc0) * avgW0 + evalHenyeyGreenstein(dot(wi, wo), mc1) * (1.f - avgW0);
    }
}
