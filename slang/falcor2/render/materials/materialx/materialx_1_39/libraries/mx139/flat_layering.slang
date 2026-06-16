// SPDX-License-Identifier: Apache-2.0

implementing mx139;

namespace mx139 {

namespace flat {

public float balance_probability(float3 albedo_a, float3 albedo_b)
{
    return mx139::balance_probability(albedo_a, albedo_b);
}

public AlbedoContributions from_scattering_albedo(float3 reflection, float3 transmission)
{
    return mx139::from_scattering_albedo(reflection, transmission);
}

// Mx139 BSDF eval_albedo follows the MaterialX 1.38 material instance contract, where
// transmission is layer pass-through and transmits[] separately controls real through-material
// transmission. OSL bsdf_mix layering uses BSDL albedo/filter_o directly, so these helpers
// reconstruct the old filter_o-compatible aggregate from codegen-known Mx139 BSDF data.
public enum LayerPassThroughAlbedoMode : int {
    // Use reflection + transmission from eval_albedo().
    scattering = 0,
    // Use reflection only; pass-through remains in filter_o transmission.
    reflection = 1,
    // Active transmissive scatter reports unity, matching BSDL microfacet albedo().
    transmissive_unity = 2,
};

public enum LayerPassThroughTransmissionMode : int {
    // Use physical eval_albedo().transmission.
    physical = 0,
    // Force an opaque filter_o pass-through.
    zero = 1,
    // Force a fully transparent filter_o pass-through.
    one = 2,
    // Add BSDL absorption to physical transmission for layer pass-through.
    absorption = 3,
    // Use absorption unless this is active transmissive scatter.
    absorption_unless_transmissive = 4,
    // Use physical transmission unless transmissive scatter should block the base.
    physical_unless_transmissive = 5,
};

public struct BsdfSummary
{
    public float3 albedo = float3(0.0);
    public float3 opacity = float3(0.0);
    public float3 reflection = float3(0.0);
    public float3 transmission = float3(0.0);
    public RoughnessInformation roughness = RoughnessInformation();
    public float3 ior = float3(0.0);
};

public struct BsdfCompatibilityAlbedo
{
    // BSDL-style albedo used by bsdf_mix branch probabilities and aggregate
    // closure albedo. This is usually reflection + transmission, but some BSDL
    // BSDFs expose a different sample albedo through albedo_impl().
    public float3 sample_albedo = float3(0.0);

    // BSDL filter_o pass-through used by layer opacity. Layer(top, base)
    // attenuates the base by this value; it is not the same contract as
    // through-material transmission exposed by a Falcor material instance.
    public float3 layer_transmission = float3(0.0);

    // Through-material transmission reconstructed for the Falcor material
    // instance contract. This is kept separate from layer_transmission because
    // OSL/BSDL layer filtering may deliberately block the base while the BSDF
    // still has physical transmission.
    public float3 material_instance_transmission = float3(0.0);
};

public BsdfSummary summarize_bsdf<TBsdf : IBSDF>(
    TBsdf bsdf,
    const float3 bsdf_weight,
    const float3 local_wi,
    const BsdfCompatibilityAlbedo compatibility_albedo
)
{
    BsdfSummary result = {};
    if (mx_hmax(bsdf_weight) <= 0.0) {
        return result;
    }
    float3 sample_albedo = compatibility_albedo.sample_albedo;
    float3 layer_transmission = compatibility_albedo.layer_transmission;
    float3 graph_transmission = min(compatibility_albedo.material_instance_transmission, sample_albedo);
    float3 graph_reflection = sample_albedo - graph_transmission;
    result.albedo = bsdf_weight * sample_albedo;
    result.opacity = saturate(bsdf_weight * saturate(float3(1.0) - layer_transmission));

    // Rebuild material-instance AlbedoContributions from BSDL sample albedo and graph-aware
    // through-material transmission.
    AlbedoContributions weighted_albedo
        = from_scattering_albedo(bsdf_weight * graph_reflection, bsdf_weight * graph_transmission);
    result.reflection = weighted_albedo.reflection;
    result.transmission = weighted_albedo.transmission;
    result.roughness = bsdf.eval_roughness(local_wi);
    result.ior = bsdf_weight * bsdf.ior_as_reflectance;
    return result;
}

float3 layer_pass_through_albedo(
    AlbedoContributions albedo,
    const LayerPassThroughAlbedoMode mode,
    const bool layer_pass_through_transmissive
)
{
    if (mode == LayerPassThroughAlbedoMode.transmissive_unity && layer_pass_through_transmissive) {
        return float3(1.0);
    }
    if (mode == LayerPassThroughAlbedoMode.reflection || mode == LayerPassThroughAlbedoMode.transmissive_unity) {
        return albedo.reflection;
    }
    return albedo.reflection + albedo.transmission;
}

float3 layer_pass_through_transmission(
    AlbedoContributions albedo,
    const LayerPassThroughTransmissionMode mode,
    const bool layer_pass_through_transmissive
)
{
    if (mode == LayerPassThroughTransmissionMode.zero) {
        return float3(0.0);
    }
    if (mode == LayerPassThroughTransmissionMode.one) {
        return float3(1.0);
    }
    if (mode == LayerPassThroughTransmissionMode.absorption_unless_transmissive && layer_pass_through_transmissive) {
        return float3(0.0);
    }
    float3 physical_transmission = albedo.transmission;
    if (mode == LayerPassThroughTransmissionMode.physical_unless_transmissive) {
        return layer_pass_through_transmissive ? float3(0.0) : physical_transmission;
    }
    if (mode == LayerPassThroughTransmissionMode.absorption
        || mode == LayerPassThroughTransmissionMode.absorption_unless_transmissive) {
        return physical_transmission + albedo.absorption_r;
    }
    return physical_transmission;
}

float3 material_instance_transmission(
    AlbedoContributions albedo,
    const LayerPassThroughAlbedoMode albedo_mode,
    const LayerPassThroughTransmissionMode transmission_mode,
    const bool layer_pass_through_transmissive
)
{
    float3 physical_transmission = albedo.transmission;
    if (transmission_mode == LayerPassThroughTransmissionMode.one) {
        return float3(1.0);
    }
    if (transmission_mode == LayerPassThroughTransmissionMode.physical_unless_transmissive) {
        return layer_pass_through_transmissive ? physical_transmission : float3(0.0);
    }
    if (albedo_mode == LayerPassThroughAlbedoMode.reflection
        || transmission_mode == LayerPassThroughTransmissionMode.zero
        || transmission_mode == LayerPassThroughTransmissionMode.absorption
        || transmission_mode == LayerPassThroughTransmissionMode.absorption_unless_transmissive) {
        return float3(0.0);
    }
    return physical_transmission;
}

// Mx139 bsdf_mix layering has to rebuild the BSDL closure-preparation view
// from Falcor-facing IBSDF data. A normal BSDF eval_albedo() tells us the
// physical reflection/transmission split used by Falcor material instances, but
// OSL/BSDL layering also uses two older BSDF contracts:
//
// - albedo_impl(): drives composite sampling mass and the closure-tree albedo.
// - filter_o(wo): drives pass-through of a layer into its base closure.
//
// Most BSDFs map directly: albedo_impl() is reflection + transmission and
// filter_o() is physical transmission. The mode enums encode the exceptions:
//
// - Transmissive dielectric/generalized-Schlick scatter uses albedo_impl() =
//   unity in BSDL, while filter_o() is zero so layer(top, base) does not show
//   the base through that transmissive scatter branch. The real transmission
//   still exists for the material instance, so it is preserved separately.
// - Reflection-only dielectric/generalized-Schlick exposes only reflection as
//   sample albedo; any pass-through sits in filter_o().
// - Sheen is reflection-only: its BSDL filter_o() represents upper-hemisphere
//   absorption, not through-material transmission.
// - Transparent/surface-unlit and simple BTDF BSDFs can force full
//   pass-through, because they are transport through the surface rather than
//   upper-layer opacity.
//
// Keeping these rules here lets generated bsdf_mix shaders ask for one opaque
// compatibility bundle instead of spelling out the BSDL-specific conversions at
// every BSDF site.
public BsdfCompatibilityAlbedo bsdf_compatibility_albedo(
    AlbedoContributions albedo,
    const LayerPassThroughAlbedoMode albedo_mode,
    const LayerPassThroughTransmissionMode transmission_mode,
    const bool layer_pass_through_transmissive
)
{
    BsdfCompatibilityAlbedo result = {};
    result.sample_albedo = layer_pass_through_albedo(albedo, albedo_mode, layer_pass_through_transmissive);
    result.layer_transmission
        = layer_pass_through_transmission(albedo, transmission_mode, layer_pass_through_transmissive);
    result.material_instance_transmission
        = material_instance_transmission(albedo, albedo_mode, transmission_mode, layer_pass_through_transmissive);
    return result;
}

public float3 eval_bsdf<TBsdf : IBSDF, S : ISampleGenerator>(
    TBsdf bsdf,
    const float3 bsdf_weight,
    Frame bsdf_frame,
    const float3 wi,
    const float3 wo,
    inout S sg,
    const BSDFContext bc
)
{
    if (mx_hmax(bsdf_weight) <= 0.0) {
        return float3(0.0);
    }
    float3 local_wi = bsdf_frame.to_local(wi);
    float3 local_wo = bsdf_frame.to_local(wo);
    if (!validate_bump_frame_hemisphere_pair(wi, local_wi, wo, local_wo)) {
        return float3(0.0);
    }
    return bsdf_weight * bsdf.eval(local_wi, local_wo, sg, bc);
}

public bool sample_bsdf<TBsdf : IBSDF, S : ISampleGenerator>(
    TBsdf bsdf,
    const float3 bsdf_weight,
    Frame bsdf_frame,
    const float3 wi,
    out float3 wo,
    out float pdf,
    out float3 sample_weight,
    out LobeTypes lobe_types,
    inout S sg,
    const BSDFContext bc
)
{
    wo = float3(0.0);
    pdf = 0.0;
    sample_weight = float3(0.0);
    lobe_types = LobeTypes::none;

    if (mx_hmax(bsdf_weight) <= 0.0) {
        return false;
    }
    float3 local_wi = bsdf_frame.to_local(wi);
    if (!validate_bump_frame_hemisphere(wi, local_wi)) {
        wo = float3(0.0);
        pdf = 0.0;
        sample_weight = float3(0.0);
        lobe_types = LobeTypes::none;
        return false;
    }
    float3 local_wo = float3(0.0);
    bool valid = bsdf.sample(local_wi, local_wo, pdf, sample_weight, lobe_types, sg, bc);
    if (!valid) {
        wo = float3(0.0);
        pdf = 0.0;
        sample_weight = float3(0.0);
        lobe_types = LobeTypes::none;
        return false;
    }
    wo = bsdf_frame.to_global(local_wo);
    if (!validate_bump_frame_hemisphere(wo, local_wo)) {
        wo = float3(0.0);
        pdf = 0.0;
        sample_weight = float3(0.0);
        lobe_types = LobeTypes::none;
        return false;
    }
    sample_weight *= bsdf_weight;
    return true;
}

public float eval_bsdf_pdf<TBsdf : IBSDF>(
    TBsdf bsdf,
    const float3 bsdf_weight,
    Frame bsdf_frame,
    const float3 wi,
    const float3 wo,
    const BSDFContext bc
)
{
    if (mx_hmax(bsdf_weight) <= 0.0) {
        return 0.0;
    }
    float3 local_wi = bsdf_frame.to_local(wi);
    float3 local_wo = bsdf_frame.to_local(wo);
    if (!validate_bump_frame_hemisphere_pair(wi, local_wi, wo, local_wo)) {
        return 0.0;
    }
    return bsdf.eval_pdf(local_wi, local_wo, bc);
}

} // namespace flat

} // namespace mx139
