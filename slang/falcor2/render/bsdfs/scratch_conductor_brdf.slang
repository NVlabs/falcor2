// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct ScratchConductorBRDF : IBSDF
{
    AnisotropicGGX D = {};
    float3 eta = {};
    float3 k = {};

    float3 scratchdirection = {};
    float depth = {};
    float3 mask = {};

    float phongCoefficient = {};

    public __init() { }

    public __init(
        AnisotropicGGX D_,
        float3 eta_,
        float3 k_,
        float3 scratchdirection_,
        float depth_,
        float3 mask_,
        float phongCoefficient_
    )
    {
        this.D = D_;
        this.eta = eta_;
        this.k = k_;
        this.scratchdirection = scratchdirection_;
        this.depth = depth_;
        this.mask = mask_;
        this.phongCoefficient = phongCoefficient_;
    }

    static const float kDefaultGrooveLength = 2000.0;
    static const float kDiracLimit = 1000000.f;

    float3 calculatePojectedIncoming(const float3 incoming, const float3 P2, const float3 P4)
    {
        // calculate projected incoming ray
        const float3 across = normalize(P2 - P4);
        const float rot = -atan2(-across.x, across.y);
        const float rotCos = cos(rot);
        const float rotSin = sin(rot);
        const float3 incomingRotAxis
            = float3(incoming.x * rotCos - incoming.y * rotSin, incoming.x * rotSin + incoming.y * rotCos, incoming.z);
        const float3 projectedIncoming0YZ = normalize(float3(0.0f, incomingRotAxis.y, incomingRotAxis.z));
        const float rotCosInv = cos(-rot);
        const float rotSinInv = sin(-rot);
        return float3(
            projectedIncoming0YZ.x * rotCosInv - projectedIncoming0YZ.y * rotSinInv,
            projectedIncoming0YZ.x * rotSinInv + projectedIncoming0YZ.y * rotCosInv,
            projectedIncoming0YZ.z
        );
    }

    bool trace(
        const float3 wiLocal,
        const float position1D,
        const float3 projectedIncomingYZ,
        out float3 outgoing,
        out float3 f
    )
    {
        outgoing = {};
        f = {};

        float3 P[6];
        float3 grooveNormal[2];

        float groovelength = kDefaultGrooveLength;
        const float cosTheta = scratchdirection[0];
        const float sinTheta = scratchdirection[1];
        P[0] = { -groovelength * sinTheta, groovelength * cosTheta, -depth };
        P[1] = { groovelength * sinTheta, -groovelength * cosTheta, -depth };
        P[2] = { -cosTheta - groovelength * sinTheta, cosTheta * groovelength - sinTheta, 0.f };
        P[3] = { -cosTheta + groovelength * sinTheta, -cosTheta * groovelength - sinTheta, 0.f };
        P[4] = { cosTheta - groovelength * sinTheta, cosTheta * groovelength + sinTheta, 0.f };
        P[5] = { cosTheta + groovelength * sinTheta, -cosTheta * groovelength + sinTheta, 0.f };

        grooveNormal[0] = normalize(cross(P[1] - P[0], -(P[2] - P[0])));
        grooveNormal[1] = normalize(cross(P[1] - P[0], P[4] - P[0]));

        float scaledPos;
        // Direction goes towards the surface.
        float3 incoming = -wiLocal;

        const float sinAngle = 1.f / sqrt(1.f + depth * depth);
        const float halfangle = asin(sinAngle);
        const float cosAngle = cos(halfangle);
        const float tanAngle = sinAngle / cosAngle;

        // Figure out how we must bias the random number to always hit a valid region. If the incoming ray is steeper
        // than the groove, we can skip this. Direction goes towards the surface.
        if (dot(projectedIncomingYZ, grooveNormal[0]) < 0.0f && dot(projectedIncomingYZ, grooveNormal[1]) < 0.0f) {
            scaledPos = position1D * 2.f - 1.f;
            scaledPos = dot(projectedIncomingYZ, grooveNormal[0]) < 0.f ? scaledPos : -scaledPos;
        } else {
            // Find out which side we can hit.
            if (dot(projectedIncomingYZ, grooveNormal[0]) < 0.0f) {
                // trace the incoming ray from the edge of the groove to see where we intersect the groove
                // this is the maximum we see from the groove, everything else is shadowed
                const float3 startpos = P[4] + (P[5] - P[4]) * 0.5f;
                const float3 p = intersectLinePlane(P[0], P[1], P[2], startpos, projectedIncomingYZ);
                const float scale = -p.z * tanAngle / (length(P[4] - P[2]) * 0.5f);
                scaledPos = -1.f + position1D * scale;
            } else if (dot(projectedIncomingYZ, grooveNormal[1]) < 0.0f) {
                const float3 startpos = P[2] + (P[3] - P[2]) * 0.5f;
                const float3 p = intersectLinePlane(P[0], P[1], P[4], startpos, projectedIncomingYZ);
                const float scale = -p.z * tanAngle / (length(P[4] - P[2]) * 0.5f);
                scaledPos = 1.f - (scale - position1D * scale);
            } else
                return false;
        }

        const float3 startpos
            = { scaledPos * scratchdirection[0],
                scaledPos * scratchdirection[1],
                -(depth - (abs(scaledPos) * 1.f / tanAngle)) };

        float4 hit = float4(startpos, scaledPos <= 0.0 ? 0.f : 1.f);
        f = float3(1.0f);

        int counter = 0;
        // we will normaly have around 1-5 bounces
        while (counter < 100) {
            incoming = normalize(reflect(incoming, grooveNormal[(int)hit[3]], f));

            const float3 hitpoint = float3(hit[0], hit[1], hit[2]);

            const float3 r1 = hitpoint;
            const float3 r2 = hitpoint + incoming;
            const float3 rDir = normalize(r2 - r1);
            const float3 left = intersectLinePlane(P[0], P[1], P[2], r1, rDir);
            const float3 right = intersectLinePlane(P[0], P[1], P[4], r1, rDir);
            hit = !checkRegion(left, P[0], P[3]) || dot(rDir, grooveNormal[0]) >= 0.f
                ? (!checkRegion(right, P[0], P[5]) ? float4(float3(0.f), -1) : float4(right, 1))
                : float4(left, 0);

            if (hit[3] < 0 || dot(-incoming, grooveNormal[(int)hit[3]]) < 0.f)
                break;

            counter++;
        }

        outgoing = incoming;
        return true;
    }

    bool checkRegion(const float3 p, const float3 pC1, const float3 pC2)
    {
        float3 a = max(pC1, pC2);
        float3 b = min(pC1, pC2);
        return p[0] <= a[0] && p[0] >= b[0] && p[1] <= a[1] && p[1] >= b[1] && p[2] <= a[2] && p[2] >= b[2];
    }

    float3 reflect(const float3 dir, const float3 n, inout float3 f)
    {
        f *= eval_fresnel_conductor(eta, k, dot(-dir, n));
        return 2.f * dot(-dir, n) * n + dir;
    }

    float3 intersectLinePlane(const float3 s1, const float3 s2, const float3 s3, const float3 r1, const float3 rDir)
    {
        // normalize to increase stability
        const float3 n = normalize(cross(s3 - s1, s2 - s1));
        return r1 + rDir * dot(r1 - s1, n) / (-dot(rDir, n));
    }

    float3 calculateEvalOrPdf(const float3 wiLocal, const float3 woLocal, const bool isEval)
    {
        // set up geometry
        float groovelength = kDefaultGrooveLength;
        float3 P[5];
        float3 grooveNormal[2];

        const float cosTheta = scratchdirection[0];
        const float sinTheta = scratchdirection[1];
        P[0] = { -groovelength * sinTheta, groovelength * cosTheta, -depth };
        P[1] = { groovelength * sinTheta, -groovelength * cosTheta, -depth };
        P[2] = { -cosTheta - groovelength * sinTheta, cosTheta * groovelength - sinTheta, 0.f };
        P[3] = { -cosTheta + groovelength * sinTheta, -cosTheta * groovelength - sinTheta, 0.f };
        P[4] = { cosTheta - groovelength * sinTheta, cosTheta * groovelength + sinTheta, 0.f };

        grooveNormal[0] = normalize(cross(P[1] - P[0], -(P[2] - P[0])));
        grooveNormal[1] = normalize(cross(P[1] - P[0], P[4] - P[0]));

        float3 incoming = -wiLocal;
        const float3 projectedIncomingYZ = -calculatePojectedIncoming(incoming, P[2], P[4]);

        // groove angle theta_v - we assume a constant width of 2; this is NOT the half angle we use above, this is the
        // FULL angle
        float thetaV = asin(1.f / sqrt(1.f + depth * depth)) * 2.f;

        float samplePosition[4];
        float sideWeight;
        int reflectionCount;
        float bounceCount[2];
        float3 a, b;
        const float2 along = normalize(float3(P[2] - P[3])).xy;
        const float sx = along.x;
        const float sy = along.y;
        const float sx2 = sx * sx;
        const float sy2 = sy * sy;

        const float cosInNLeft = dot(projectedIncomingYZ, grooveNormal[0]);
        const float cosInNRight = dot(projectedIncomingYZ, grooveNormal[1]);

        // If we can see both sides, we can have a maximum of four different rays. Since we draw a random number from
        // the whole range, we can partition the groove symetrically and look at both sides individually
        if (cosInNLeft > 0.f && cosInNRight > 0.f) {
            // we see both sides; the max number of reflections is 4
            samplePosition = { 0.01f, 0.499f, 0.501f, 0.99f };
            const float acosYZ = acos(projectedIncomingYZ.z);
            // number of bounces k; we have either k or k - 1 bounces; we need to calculate it for both sides
            if (cosInNLeft > cosInNRight) {
                bounceCount[0] = floor((float.PI + (-acosYZ)) / thetaV + 0.5f);
                bounceCount[1] = -floor((float.PI + acosYZ) / thetaV + 0.5f);
            } else {
                bounceCount[0] = floor((float.PI + acosYZ) / thetaV + 0.5f);
                bounceCount[1] = -floor((float.PI + (-acosYZ)) / thetaV + 0.5f);
            }

            a = P[4];
            b = P[2];

            // both sides are equally visible, split the weight
            sideWeight = 0.5f;
            reflectionCount = 2;
        } else {
            // we see only one side; the maximum number of reflections is 2
            // clamping has to be conistent with values in sample()
            samplePosition = { 0.01f, 0.99f, -1.f, -1.f };

            // switch rotation points depending on the whch side is hit
            if (cosInNLeft < 0.f) {
                // number of bounces k; we have either k or k - 1 bounces;
                bounceCount[0] = -(floor((float.PI + 2.f * (-acos(projectedIncomingYZ.z))) / thetaV) + 1.f);
                a = P[2];
                b = P[4];
            } else {
                // number of bounces k; we have either k or k - 1 bounces;
                bounceCount[0] = (floor((float.PI + 2.f * (-acos(projectedIncomingYZ.z))) / thetaV)) + 1.f;
                a = P[4];
                b = P[2];
            }

            sideWeight = 1.f;
            reflectionCount = 1;
        }

        float3 evalScratch = 0.f;
        for (int i = 0; i < reflectionCount; i++) {
            const float3 aPivot = a - P[0];
            const float c = cos(-thetaV * bounceCount[i]);
            const float s = sin(-thetaV * bounceCount[i]);

            const float3 kPoint
                = { P[0].x - aPivot.y * sx * sy * (-1 + c) + aPivot.x * (sx * sx + c - sx * sx * c) + aPivot.z * sy * s,
                    P[0].y - aPivot.x * sx * sy * (-1 + c) + aPivot.y * (sy * sy + c - sy * sy * c)
                        + (-a.z + P[0].z) * sx * s,
                    P[0].z + aPivot.z * c + aPivot.y * sx * s - aPivot.x * sy * s };

            // project k on the plane between a and b
            const float3 kProjected = intersectLinePlane(P[2], P[3], P[4], kPoint, projectedIncomingYZ);
            const float distAcross = length(a - b);
            float distA = length(a - kProjected);
            float distB = length(b - kProjected);

            // sanity check: remove invalid points; this can happen if there is only one reflection instead of two
            distA = distA > distAcross ? 0.0f : distA;
            distB = distB > distAcross ? 0.0f : distB;

            // sanity check: only one of them should be zero, never both unless we have an error in the bounceCount
            float r;
            if (distA > 0.0f && distB <= 0.f)
                r = 1.f;
            else if (distA <= 0.0f && distB > 0.f)
                r = 0.f;
            else if (distA > 0.f && distB > 0.f)
                r = 1.f - distA / (distA + distB);

            const float areaRatio[2] = { r, 1.f - r };

            for (int j = 0; j < 2; j++) {
                float pos = samplePosition[i * 2 + j];
                float3 outgoing;
                float3 f = float3(1.f);
                bool valid = trace(wiLocal, pos, -projectedIncomingYZ, outgoing, f);
                // for the PDF, we need a different estimator; normalissation is done in evalPdf
                f = isEval ? f : 1.f;
                const float exponent = isEval ? phongCoefficient : phongCoefficient + 1.f;

                evalScratch
                    += (valid ? f * areaRatio[j] * pow(max(dot(outgoing, woLocal), 0.f), phongCoefficient) * woLocal.z
                                * sideWeight
                              : 0.0f);
            }

            const float3 temp = a;
            a = b;
            b = temp;
        }

        // safety precaution
        if (any(isnan(evalScratch)) || any(isinf(evalScratch)))
            evalScratch = 0.0f;

        return evalScratch;
    }

    public float3 eval<S : ISampleGenerator>(const float3 wiLocal, const float3 woLocal, inout S sg, BSDFContext bc)
    {
        if (min(wiLocal.z, woLocal.z) < BSDF_MIN_COS_THETA)
            return float3(0.f);

        // eval and pdf have similar computations, but we remove the Fresnel factor for the PDF
        float3 evalScratch = 0.f;

        if (phongCoefficient < kDiracLimit)
            evalScratch = calculateEvalOrPdf(wiLocal, woLocal, true);

        PBRTConductorBSDF unscratched = { D, eta, k };
        return unscratched.eval(wiLocal, woLocal, sg, bc) * (1.0 - mask[0])
            + evalScratch * mask[0] * (phongCoefficient + 2.f) / (2.f * float.PI);
    }

    public bool sample<S : ISampleGenerator>(
        float3 wiLocal,
        out float3 wo,
        out float pdf,
        out float3 weight,
        out LobeTypes lobeTypes,
        inout S sg,
        BSDFContext bc = {}
    )
    {
        wo = {};
        pdf = {};
        weight = {};
        lobeTypes = {};

        if (wiLocal.z < 0.0f)
            return false;

        const float2 rand = next_float2(sg);
        const float threshold = rand[0];

        // eval() and eval_pdf() expose the full masked mixture of the unscratched conductor and scratch lobes.
        // sample() chooses one branch and returns that branch's local pdf/weight, so statistical tests that
        // expect sample().pdf == eval_pdf() and sample().weight == eval()/eval_pdf() do not apply here.
        bool valid = false;
        float3 outgoing;
        float3 f;
        if (mask[0] <= threshold) {
            PBRTConductorBSDF unscratched = { D, eta, k };
            return unscratched.sample(wiLocal, wo, pdf, weight, lobeTypes, sg, bc);
        } else {
            const float groovelength = kDefaultGrooveLength;
            const float cosTheta = scratchdirection[0];
            const float sinTheta = scratchdirection[1];
            const float3 P2 = { -cosTheta - groovelength * sinTheta, cosTheta * groovelength - sinTheta, 0.f };
            const float3 P4 = { cosTheta - groovelength * sinTheta, cosTheta * groovelength + sinTheta, 0.f };

            // note that we have a different sign here
            const float3 projectedIncomingYZ = calculatePojectedIncoming(-wiLocal, P2, P4);
            // avoid nummerical precision issues with rescaling
            const float pos = 0.01f + rand[1] * 0.99f;
            valid = trace(wiLocal, pos, projectedIncomingYZ, outgoing, f);

            if (valid) {
                // We apply the roughness after the bounces. This does not simulate rough grooves but allows us to
                // perform eval() and evalPdf(). While it's possible to set the roughness really low, a good value
                // is between 10k and 50k.
                const float2 randPhong = next_float2(sg);
                const float theta = acos(pow(randPhong[0], 1.f / (phongCoefficient + 1.f)));
                const float phi = 2.f * float.PI * randPhong[1];

                const float3 phongLocal = spherical_to_cartesian(float2(theta, phi));

                Frame sf = Frame(outgoing);

                wo = sf.to_global(phongLocal);

                // If the roughness is too high, a lot of rays will go into the surface. The only way to prevent this is
                // to do the roughening during the bounces and reflect rays back, but then we won't have a deterministic
                // eval and pdf
                if (min(wiLocal.z, wo.z) < BSDF_MIN_COS_THETA) {
                    weight = {};
                    return false;
                }

                // switch to Dirac
                if (phongCoefficient < kDiracLimit) {
                    const float evalScratch = calculateEvalOrPdf(wiLocal, wo, false)[0];
                    pdf = evalScratch * (phongCoefficient + 2.f) / (2.f * float.PI);

                    // This should not happen. But it occasionally does in regions where we blend from scratch to
                    // no-scratch if the scratch direction doesn't have enough bleed.
                    if (pdf <= 0.f) {
                        weight = {};
                        return false;
                    }

                    weight = f;
                    lobeTypes = LobeTypes::non_delta_reflection;
                } else {
                    weight = f;
                    pdf = 1.f;
                    lobeTypes = LobeTypes::delta_reflection;
                }
            }
        }

        return valid;
    }

    public float eval_pdf(const float3 wiLocal, const float3 woLocal, BSDFContext bc)
    {
        if (min(wiLocal.z, woLocal.z) < BSDF_MIN_COS_THETA)
            return 0.f;

        // eval and pdf have similar computations, but we remove the Fresnel factor for the PDF
        // the PDF is wavelength independent
        float evalScratch = 0.f;

        if (phongCoefficient < kDiracLimit)
            evalScratch = calculateEvalOrPdf(wiLocal, woLocal, false)[0];

        PBRTConductorBSDF unscratched = { D, eta, k };
        return unscratched.eval_pdf(wiLocal, woLocal, bc) * (1.0 - mask[0])
            + evalScratch * mask[0] * (phongCoefficient + 2.f) / (2.f * float.PI);
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float3 r = eval_fresnel_conductor(eta, k, wi.z);
        return AlbedoContributions(r, 1.0f - r, 0.0f, 0.0f);
    }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(1.f);
        }
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        RoughnessInformation r = {};
        r.roughness = D.alpha;
        return r;
    }
};
