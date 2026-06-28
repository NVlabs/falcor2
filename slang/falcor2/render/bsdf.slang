// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;

/** Low-level interface for BSDF functions.

    Conventions:
    - All operations are done in a local coordinate frame.
    - The local frame has normal N=(0,0,1), tangent T=(1,0,0) and bitangent B=(0,1,0).
    - The incident and outgoing direction point away from the shading location.
    - For surface BSDFs, the incident direction (wi) is always in the positive hemisphere.
    - Curve BSDFs, such as hair scattering in a curve frame, may evaluate and sample over the full sphere.
    - The outgoing direction (wo) is sampled.
    - Surface BSDF eval() returns f(wi, wo) * abs(dot(wo, n)). Curve BSDF eval() returns the model
      scattering response and does not apply an extra surface foreshortening term. Curve-specific measure
      terms and Jacobians belong to the BSDF and its pdf
      (see https://pbr-book.org/4ed/Reflection_Models/Scattering_from_Hair).
*/

/// Minimum cos(theta) for the incident and outgoing vectors.
/// Some BSDF functions are not robust for cos(theta) == 0.0,
/// so using a small epsilon for consistency.
public static constexpr float BSDF_MIN_COS_THETA = 1e-6;

public enum LobeTypes : uint {
    none = 0x00,

    diffuse_reflection = 0x01,
    glossy_reflection = 0x02,
    delta_reflection = 0x04,

    diffuse_transmission = 0x10,
    glossy_transmission = 0x20,
    delta_transmission = 0x40,

    glossy_curve = 0x80,

    diffuse = diffuse_reflection | diffuse_transmission,
    glossy = glossy_reflection | glossy_transmission,
    delta = delta_reflection | delta_transmission,
    curve = glossy_curve,

    reflection = diffuse_reflection | glossy_reflection | delta_reflection,
    transmission = diffuse_transmission | glossy_transmission | delta_transmission,

    non_delta = diffuse | glossy | curve,
    non_delta_reflection = diffuse_reflection | glossy_reflection,
    non_delta_transmission = diffuse_transmission | glossy_transmission,

    all = reflection | transmission | curve,
};

/// Albedo is split into four components. They have to sum up to 1.0.
public struct AlbedoContributions
{
    /// Energy reflected into upper hemisphere.
    public float3 reflection;
    /// Energy absorbed in upper hemisphere.
    public float3 absorption_r;
    /// Energy transmitted into lower hemisphere.
    public float3 transmission;
    /// Energy absorbed in lower hemisphere.
    public float3 absorption_t;

    public __init(float3 reflection, float3 absorption_r, float3 transmission, float3 absorption_t)
    {
        this.reflection = reflection;
        this.absorption_r = absorption_r;
        this.transmission = transmission;
        this.absorption_t = absorption_t;
    }
};

AlbedoContributions operator +(const AlbedoContributions a, const AlbedoContributions b)
{
    return AlbedoContributions(
        a.reflection + b.reflection,
        a.absorption_r + b.absorption_r,
        a.transmission + b.transmission,
        a.absorption_t + b.absorption_t
    );
}

AlbedoContributions operator +=(AlbedoContributions a, const AlbedoContributions b)
{
    a.reflection += b.reflection;
    a.absorption_r += b.absorption_r;
    a.transmission += b.transmission;
    a.absorption_t += b.absorption_t;

    return a;
}

AlbedoContributions operator*(const AlbedoContributions a, const AlbedoContributions b)
{
    return AlbedoContributions(
        a.reflection * b.reflection,
        a.absorption_r * b.absorption_r,
        a.transmission * b.transmission,
        a.absorption_t * b.absorption_t
    );
}

AlbedoContributions operator*(const float f, const AlbedoContributions b)
{
    return AlbedoContributions(f * b.reflection, f * b.absorption_r, f * b.transmission, f * b.absorption_t);
}

public struct RoughnessInformation
{
    /// The (mostly meaningless) non-universal roughness of the BSDF.
    public float2 roughness;
    public float3 scratch;

    public __init()
    {
        roughness = float2(0.f);
        scratch = float3(0.f);
    }

    public __init(float2 roughness, float3 scratch)
    {
        this.roughness = roughness;
        this.scratch = scratch;
    }
}

public struct BSDFContext
{
    /// Index of refraction from incidence medium.
    public float ior_i;
    /// Index of refraction from transmission medium.
    public float ior_t;
    /// Soft sampling-efficiency hint from a higher-level material.
    ///
    /// The caller expects sampled lobes outside this mask to be discarded by
    /// surrounding material logic, so a BSDF may avoid sampling them and report
    /// the matching hinted sampling PDF. Ignoring this hint is still correct
    /// BSDF behavior; it only spends samples on events the caller will later
    /// reject.
    public LobeTypes sample_lobe_types_hint;

    public __init()
    {
        ior_i = 1.0;
        ior_t = 1.0;
        sample_lobe_types_hint = LobeTypes::all;
    }

    public __init(float ior_i, float ior_t)
    {
        this.ior_i = ior_i;
        this.ior_t = ior_t;
        sample_lobe_types_hint = LobeTypes::all;
    }

    public __init(float ior_i, float ior_t, LobeTypes sample_lobe_types_hint)
    {
        this.ior_i = ior_i;
        this.ior_t = ior_t;
        this.sample_lobe_types_hint = sample_lobe_types_hint;
    }
};

public interface IBSDF
{
    /// Evaluates the BSDF.
    /// @param wi Incident direction.
    /// @param wo Outgoing direction.
    /// @param[inout] sg Sample generator.
    /// @param bc BSDF context.
    /// @return For surface BSDFs, returns f(wi, wo) * abs(dot(wo, n)). Curve BSDFs return their scattering
    /// response without an extra surface foreshortening factor.
    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc);

    /// Samples the BSDF.
    /// @param wi Incident direction.
    /// @param[out] wo Outgoing direction.
    /// @param[out] pdf pdf with respect to solid angle for sampling outgoing direction wo (0 for delta events).
    /// @param[out] weight Sample weight eval(wi, wo) / pdf(wo).
    /// @param[out] lobe_types Sampled lobe types (see LobeTypes).
    /// @param[inout] sg Sample generator.
    /// @param bc BSDF context.
    /// @return Returns true if successful.
    public bool sample<S : ISampleGenerator>(
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 weight,
        out LobeTypes lobe_types,
        inout S sg,
        const BSDFContext bc
    );

    /// Evaluates the directional pdf for sampling outgoing direction wo.
    /// @param wi Incident direction.
    /// @param wo Outgoing direction.
    /// @param bc BSDF context.
    /// @return Returns the pdf with respect to solid angle for sampling outgoing direction wo (0 for delta events).
    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc);

    /// Evaluate albedos (hemispherical reflectance) of the BSDF. Reflection + transmission hemisphere should be <= 1.0.
    /// Curve BSDFs may report an approximate full-sphere scattering albedo.
    /// @param wi Incident direction.
    /// @return Returns the albedo contributions.
    public AlbedoContributions eval_albedo(const float3 wi);

    /// Information about roughness of the BSDF in (various) forms.
    /// @param wi Incident direction.
    /// @return Returns the roughness.
    public RoughnessInformation eval_roughness(const float3 wi);

    /// Unattenuated reflectance at normal incidence primarily used to get a common-form IOR.
    public property float3 ior_as_reflectance { get; }
};
