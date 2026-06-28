// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Unrefracted dirac transmission
/// The transmission color can be used to tint
public struct SimpleBTDF : IBSDF
{
    ///< Transmittance color (i.e. tint).
    float3 transmittance = {};

    public __init() { }

    public __init(float3 transmittance) { this.transmittance = transmittance; }

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return float3(0.0);
    }

    public bool sample<S : ISampleGenerator>(
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 weight,
        out LobeTypes lobe_types,
        inout S sg,
        const BSDFContext bc
    )
    {
        // Default initialization to avoid divergence at returns.
        wo = {};
        weight = {};
        pdf = {};
        lobe_types = {};

        if (wi.z > BSDF_MIN_COS_THETA)
            return false;

        wo = float3(-wi.x, -wi.y, -wi.z);
        pdf = 0.0;
        weight = transmittance;
        lobe_types = LobeTypes::delta_transmission;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc) { return 0.0; }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        return AlbedoContributions(0.0, 0.0, transmittance, 1.0 - transmittance);
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(0.0), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }
};
