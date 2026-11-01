// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

// Preserve Standard's smooth-lobe approximation during extraction. This is not
// a glTF requirement: alpha below this value is represented by a delta event.
public static const float GLTF_MIN_GGX_ALPHA = 0.0064;

public struct GLTFAnisotropyInputs
{
    public float strength = 0.0;
    public float2 direction = float2(1.0, 0.0);
};

public struct GLTFDiffuseTransmissionInputs
{
    public float factor = 0.0;
    public float3 color = float3(1.0);
};

public struct GLTFSheenInputs
{
    public float3 color = float3(0.0);
    public float roughness = 0.0;
};

public struct GLTFIridescenceInputs
{
    public float factor = 0.0;
    public float ior = 1.3;
    public float thickness = 400.0;
};

public struct GLTFTransmissionInputs
{
    public float factor = 0.0;
};

public struct GLTFBSDFData<let Features : uint>
{
    public static const bool HAS_ANISOTROPY = (Features & gltfmat::FeatureBits::ANISOTROPY) != 0u;
    public static const bool HAS_DIFFUSE_TRANSMISSION = (Features & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u;
    public static const bool HAS_SHEEN = (Features & gltfmat::FeatureBits::SHEEN) != 0u;
    public static const bool HAS_IRIDESCENCE = (Features & gltfmat::FeatureBits::IRIDESCENCE) != 0u;
    public static const bool HAS_TRANSMISSION = (Features & gltfmat::FeatureBits::TRANSMISSION) != 0u;

    // Resolved linear parameters; GLTFMaterial handles texture lookup and color decoding.
    public float3 base_color = {};
    public float metallic = {};
    public float roughness = {};
    public Conditional<GLTFAnisotropyInputs, HAS_ANISOTROPY> anisotropy
        = Conditional<GLTFAnisotropyInputs, HAS_ANISOTROPY>(GLTFAnisotropyInputs());
    public Conditional<GLTFDiffuseTransmissionInputs, HAS_DIFFUSE_TRANSMISSION> diffuse_transmission
        = Conditional<GLTFDiffuseTransmissionInputs, HAS_DIFFUSE_TRANSMISSION>(GLTFDiffuseTransmissionInputs());
    public Conditional<GLTFSheenInputs, HAS_SHEEN> sheen = Conditional<GLTFSheenInputs, HAS_SHEEN>(GLTFSheenInputs());
    public Conditional<GLTFIridescenceInputs, HAS_IRIDESCENCE> iridescence
        = Conditional<GLTFIridescenceInputs, HAS_IRIDESCENCE>(GLTFIridescenceInputs());

    // Dielectric endpoints already include the specular extension's tint and strength.
    public float3 dielectric_f0 = float3(0.04);
    public float dielectric_f90 = 1.0;
    public Conditional<GLTFTransmissionInputs, HAS_TRANSMISSION> transmission
        = Conditional<GLTFTransmissionInputs, HAS_TRANSMISSION>(GLTFTransmissionInputs());
    public float ior = 1.5;

    // Incident/transmitted IOR ratio for bulk refraction; distinct from the material's absolute IOR.
    public float eta = 1.0;
    public bool thin_walled = true;
};

public struct GLTFAnisotropyState
{
    public float2 direction = float2(1.0, 0.0);
    public AnisotropicGGXPolicy<10000, true> distribution = {};
    public bool active = false;
};

public struct GLTFDiffuseTransmissionState
{
    public LambertDiffuseBRDF lobe = {};
    public float proposal_probability = 0.0;
};

public struct GLTFSheenState
{
    public float3 color = float3(0.0);
    public float roughness = 0.0;
    public float proposal_probability = 0.0;
    public float incident_energy = 0.0;
};

public struct GLTFIridescenceState
{
    public float factor = 0.0;
    public float ior = 1.3;
    public float thickness = 400.0;
};

public struct GLTFTransmissionState
{
    public ThinSpecularMicrofacetBTDF thin = {};
    public RefractiveSpecularMicrofacetBTDF bulk = {};
    public float3 tint = float3(0.0);
    public float proposal_probability = 0.0;
};

public struct GLTFMultiscatterState
{
    public GGXMultipleScatteringBRDF brdf = {};
    public float weight = 0.0;
    public float proposal_probability = 0.0;
};

/// glTF metallic-roughness composition with thin-sheet or refractive transmission.
/// Values include the outgoing cosine. Clearcoat is composed separately by GLTFMaterialInstance.
/// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#material-structure
public struct GLTFBSDF<let Features : uint> : IBSDF
{
    public static const bool HAS_DIFFUSE_TRANSMISSION = (Features & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u;
    public static const bool HAS_ANISOTROPY = (Features & gltfmat::FeatureBits::ANISOTROPY) != 0u;
    public static const bool HAS_TRANSMISSION = (Features & gltfmat::FeatureBits::TRANSMISSION) != 0u;
    public static const bool HAS_MULTISCATTER = (Features & gltfmat::FeatureBits::MULTISCATTER) != 0u;
    public static const bool HAS_SHEEN = (Features & gltfmat::FeatureBits::SHEEN) != 0u;
    public static const bool HAS_IRIDESCENCE = (Features & gltfmat::FeatureBits::IRIDESCENCE) != 0u;

    public uint feature_mask = 0u;
    public LambertDiffuseBRDF diffuse = {};
    public Conditional<GLTFDiffuseTransmissionState, HAS_DIFFUSE_TRANSMISSION> diffuse_transmission
        = Conditional<GLTFDiffuseTransmissionState, HAS_DIFFUSE_TRANSMISSION>(GLTFDiffuseTransmissionState());
    public Conditional<GLTFAnisotropyState, HAS_ANISOTROPY> anisotropy
        = Conditional<GLTFAnisotropyState, HAS_ANISOTROPY>(GLTFAnisotropyState());
    public SpecularMicrofacetBRDF specular = {};
    public Conditional<GLTFTransmissionState, HAS_TRANSMISSION> transmission
        = Conditional<GLTFTransmissionState, HAS_TRANSMISSION>(GLTFTransmissionState());
    public bool thin_walled = true;
    public float eta = 1.0;
    public float metallic = {};
    public float3 metal_f0 = {};
    public Conditional<GLTFMultiscatterState, HAS_MULTISCATTER> multiscatter
        = Conditional<GLTFMultiscatterState, HAS_MULTISCATTER>(GLTFMultiscatterState());
    public float specular_weight = 1.0;
    public Conditional<GLTFSheenState, HAS_SHEEN> sheen = Conditional<GLTFSheenState, HAS_SHEEN>(GLTFSheenState());
    public Conditional<GLTFIridescenceState, HAS_IRIDESCENCE> iridescence
        = Conditional<GLTFIridescenceState, HAS_IRIDESCENCE>(GLTFIridescenceState());
    public float3 dielectric_f0 = float3(0.04);
    public float dielectric_f90 = 1.0;
    public float p_diffuse = {};
    public float p_specular = {};
    public BSDFFlags lobe_types = BSDFFlags::none;

    public __init() { }

    public __init(const float3 wi, const GLTFBSDFData<Features> data, uint feature_mask)
    {
        // The compile mask determines storage. This one runtime mask determines
        // which of the compiled lobes participate at the current material hit.
        this.feature_mask = feature_mask;
        float metallic = saturate(data.metallic);
        this.metallic = metallic;
        metal_f0 = data.base_color;
        thin_walled = data.thin_walled;
        eta = data.eta;

        float anisotropy_strength = 0.0;
        bool anisotropic = false;
        if (let inputs = data.anisotropy.get()) {
            if ((feature_mask & gltfmat::FeatureBits::ANISOTROPY) != 0u) {
                anisotropy_strength = saturate(inputs.strength);
                anisotropic = anisotropy_strength > 0.0;
                GLTFAnisotropyState state = {};
                state.active = anisotropic;
                state.direction = inputs.direction;
                // glTF broadens tangent GGX by anisotropy^2; the other width stays r^2.
                const float alpha = saturate(data.roughness) * saturate(data.roughness);
                state.distribution = AnisotropicGGXPolicy<10000, true>(
                    float2(lerp(alpha, 1.0, anisotropy_strength * anisotropy_strength), alpha)
                );
                anisotropy.set(state);
            }
        }

        float iridescence_factor = 0.0;
        if (let inputs = data.iridescence.get()) {
            if ((feature_mask & gltfmat::FeatureBits::IRIDESCENCE) != 0u) {
                GLTFIridescenceState state = {};
                state.factor = inputs.thickness > 0.0 ? saturate(inputs.factor) : 0.0;
                state.ior = inputs.ior;
                state.thickness = inputs.thickness;
                iridescence_factor = state.factor;
                iridescence.set(state);
            }
        }

        float p_sheen = 0.0;
        if (let inputs = data.sheen.get()) {
            if ((feature_mask & gltfmat::FeatureBits::SHEEN) != 0u) {
                GLTFSheenState state = {};
                state.color = saturate(inputs.color);
                state.roughness = saturate(inputs.roughness);
                // The table integrates unit-color sheen. The floor supports its proposal.
                if (any(state.color > 0.0)) {
                    state.incident_energy = gltf_sheen::energy(wi.z, state.roughness);
                    state.proposal_probability = max(0.01, luminance(state.color) * state.incident_energy);
                }
                p_sheen = state.proposal_probability;
                sheen.set(state);
            }
        }

        float roughness = saturate(data.roughness);
        dielectric_f0 = saturate(data.dielectric_f0);
        dielectric_f90 = saturate(data.dielectric_f90);
        float alpha = roughness * roughness;
        bool is_delta = !anisotropic && alpha < GLTF_MIN_GGX_ALPHA;
        if (alpha < GLTF_MIN_GGX_ALPHA)
            alpha = 0.0;

        float T = 0.0;
        if (let inputs = data.transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u)
                T = saturate(inputs.factor);
        }
        float D = 0.0;
        float3 diffuse_transmission_color = float3(1.0);
        if (let inputs = data.diffuse_transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u) {
                D = saturate(inputs.factor);
                diffuse_transmission_color = inputs.color;
            }
        }

        // Nested mixing: (1-m)(1-T)(1-D), (1-m)(1-T)D, (1-m)T.
        diffuse = LambertDiffuseBRDF(data.base_color * (1.0 - metallic) * (1.0 - T) * (1.0 - D));
        float p_diffuse_transmission = 0.0;
        if (let inputs = data.diffuse_transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u) {
                GLTFDiffuseTransmissionState state = {};
                state.lobe = LambertDiffuseBRDF(diffuse_transmission_color * (1.0 - metallic) * (1.0 - T) * D);
                state.proposal_probability = luminance(state.lobe.albedo);
                p_diffuse_transmission = state.proposal_probability;
                diffuse_transmission.set(state);
            }
        }

        float p_transmission = 0.0;
        if (let inputs = data.transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u) {
                GLTFTransmissionState state = {};
                // The IOR heuristic narrows perceptual roughness before squaring.
                float transmission_roughness = roughness * (data.ior == 0.0 ? 1.0 : saturate(2.0 * data.ior - 2.0));
                float transmission_alpha = transmission_roughness * transmission_roughness;
                if (transmission_alpha < GLTF_MIN_GGX_ALPHA)
                    transmission_alpha = 0.0;
                state.thin = ThinSpecularMicrofacetBTDF(transmission_alpha);
                state.bulk = RefractiveSpecularMicrofacetBTDF(alpha, eta);
                state.tint = data.base_color * (1.0 - metallic) * T;
                state.proposal_probability = luminance(state.tint);
                p_transmission = state.proposal_probability;
                transmission.set(state);
            }
        }

        // Compensate only the ordinary metallic share. The conditional removes
        // both prepared state and table access when MULTISCATTER is compile-off.
        float p_multiscatter = 0.0;
        if (let state = multiscatter.get()) {
            if ((feature_mask & gltfmat::FeatureBits::MULTISCATTER) != 0u && !is_delta && metallic > 0.0
                && iridescence_factor < 1.0) {
                GLTFMultiscatterState prepared = state;
                float compensation_roughness = roughness;
                if (let anisotropy_state = anisotropy.get()) {
                    if ((feature_mask & gltfmat::FeatureBits::ANISOTROPY) != 0u && anisotropy_state.active) {
                        const float2 widths = anisotropy_state.distribution.alpha;
                        compensation_roughness = sqrt(sqrt(widths.x * widths.y));
                    }
                }
                prepared.brdf = GGXMultipleScatteringBRDF(wi, compensation_roughness, data.base_color);
                prepared.weight = metallic * (1.0 - iridescence_factor);
                prepared.proposal_probability = prepared.weight * luminance(prepared.brdf.eval_albedo(wi).reflection);
                p_multiscatter = prepared.proposal_probability;
                multiscatter.set(prepared);
            }
        }

        // Schlick is linear in F0 and F90. The unit-grazing GGX leaf is shared.
        specular_weight = lerp(dielectric_f90, 1.0, metallic);
        float3 specular_f0 = lerp(dielectric_f0, data.base_color, metallic);
        specular = SpecularMicrofacetBRDF(
            specular_weight > 0.0 ? specular_f0 / specular_weight : float3(0.0),
            alpha,
            BSDFFlags::all
        );

        if (!thin_walled || iridescence_factor > 0.0 || anisotropic) {
            specular.albedo = float3(1.0);
            specular_weight = 1.0;
        }

        lobe_types = is_delta ? BSDFFlags::delta_reflection : BSDFFlags::glossy_reflection;
        if (p_sheen > 0.0)
            lobe_types |= BSDFFlags::glossy_reflection;
        if (any(diffuse.albedo > 0.0))
            lobe_types = BSDFFlags(uint(lobe_types) | uint(BSDFFlags::diffuse_reflection));
        if (let state = diffuse_transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u && any(state.lobe.albedo > 0.0))
                lobe_types |= BSDFFlags::diffuse_transmission;
        }
        if (let state = transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u && any(state.tint > 0.0))
                lobe_types = BSDFFlags(
                    uint(lobe_types)
                    | uint(
                        (thin_walled ? state.thin.alpha : state.bulk.alpha) == 0.0 ? BSDFFlags::delta_transmission
                                                                                   : BSDFFlags::glossy_transmission
                    )
                );
        }

        // Preserve the baseline proposal weights and random sequence.
        p_diffuse = luminance(diffuse.albedo) * (1.0 - metallic);
        p_specular = specular_weight * luminance(eval_fresnel_schlick(specular.albedo, float3(1.0), wi.z));
        if (!thin_walled || iridescence_factor > 0.0 || anisotropic)
            p_specular = max(0.01, luminance(reflection_fresnel(wi.z)));
        if (p_specular == 0.0 && specular_weight > 0.0)
            p_specular = 0.01 * specular_weight;

        if (let state = sheen.get()) {
            if ((feature_mask & gltfmat::FeatureBits::SHEEN) != 0u && state.proposal_probability > 0.0) {
                const float scale
                    = max(0.01, 1.0 - max(state.color.x, max(state.color.y, state.color.z)) * state.incident_energy);
                p_diffuse *= scale;
                p_diffuse_transmission *= scale;
                p_specular *= scale;
                p_transmission *= scale;
                p_multiscatter *= scale;
            }
        }

        float norm = p_diffuse + p_specular + p_transmission + p_multiscatter + p_sheen + p_diffuse_transmission;
        if (norm > 0.0) {
            norm = 1.0 / norm;
            p_diffuse *= norm;
            p_specular *= norm;
            p_transmission *= norm;
            p_multiscatter *= norm;
            p_sheen *= norm;
            p_diffuse_transmission *= norm;
        }
        if (let state = transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u) {
                GLTFTransmissionState normalized = state;
                normalized.proposal_probability = p_transmission;
                transmission.set(normalized);
            }
        }
        if (let state = diffuse_transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u) {
                GLTFDiffuseTransmissionState normalized = state;
                normalized.proposal_probability = p_diffuse_transmission;
                diffuse_transmission.set(normalized);
            }
        }
        if (let state = sheen.get()) {
            if ((feature_mask & gltfmat::FeatureBits::SHEEN) != 0u) {
                GLTFSheenState normalized = state;
                normalized.proposal_probability = p_sheen;
                sheen.set(normalized);
            }
        }
        if (let state = multiscatter.get()) {
            if ((feature_mask & gltfmat::FeatureBits::MULTISCATTER) != 0u) {
                GLTFMultiscatterState normalized = state;
                normalized.proposal_probability = p_multiscatter;
                multiscatter.set(normalized);
            }
        }
    }

    public BSDFFlags get_lobe_types() { return lobe_types; }

    private bool has_anisotropy()
    {
        if (let state = anisotropy.get()) {
            if ((feature_mask & gltfmat::FeatureBits::ANISOTROPY) != 0u)
                return state.active;
        }
        return false;
    }

    private float iridescence_factor()
    {
        if (let state = iridescence.get()) {
            if ((feature_mask & gltfmat::FeatureBits::IRIDESCENCE) != 0u)
                return state.factor;
        }
        return 0.0;
    }

    private float3 dielectric_fresnel(float cosine)
    {
        // eta belongs to the core: bulk dielectric reflection can use it even
        // when the TRANSMISSION lobe is compiled away.
        if (!thin_walled && eta != 1.0) {
            float ct2 = microfacet_refraction::transmitted_cosine_squared(cosine, eta);
            if (ct2 <= 0.0)
                return float3(1.0);
            if (eta > 1.0)
                cosine = sqrt(ct2);
        }

        float3 f = !thin_walled && eta == 1.0 ? float3(0.0)
                                              : eval_fresnel_schlick(dielectric_f0, float3(dielectric_f90), cosine);
        if (let state = iridescence.get()) {
            if ((feature_mask & gltfmat::FeatureBits::IRIDESCENCE) != 0u && state.factor > 0.0)
                f = lerp(f, gltf_iridescence::fresnel(cosine, state.ior, state.thickness, dielectric_f0), state.factor);
        }
        return f;
    }

    private float3 reflection_fresnel(float cosine)
    {
        float3 metal = eval_fresnel_schlick(metal_f0, float3(1.0), cosine);
        if (let state = iridescence.get()) {
            if ((feature_mask & gltfmat::FeatureBits::IRIDESCENCE) != 0u && state.factor > 0.0)
                metal = lerp(
                    metal,
                    gltf_iridescence::fresnel(cosine, state.ior, state.thickness, metal_f0),
                    state.factor
                );
        }
        return lerp(dielectric_fresnel(cosine), metal, metallic);
    }

    private float sheen_base_weight(float3 wo)
    {
        if (let state = sheen.get()) {
            if ((feature_mask & gltfmat::FeatureBits::SHEEN) != 0u && state.proposal_probability > 0.0) {
                // Symmetric attenuation uses the larger directional sheen energy.
                const float outgoing_energy = gltf_sheen::energy(wo.z, state.roughness);
                return max(
                    0.0,
                    1.0
                        - max(state.color.x, max(state.color.y, state.color.z))
                            * max(state.incident_energy, outgoing_energy)
                );
            }
        }
        return 1.0;
    }

    private float3 specular_layer_weight(const float3 wi, const float3 wo)
    {
        return (thin_walled && iridescence_factor() <= 0.0 && !has_anisotropy()) || wo.z <= 0.0
            ? float3(specular_weight)
            : reflection_fresnel(saturate(dot(wi, normalize(wi + wo))));
    }

    private float diffuse_layer_weight(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return 0.0;
        float wi_dot_h = saturate(dot(wi, normalize(wi + wo)));
        float3 f = dielectric_fresnel(wi_dot_h);
        return max(0.0, 1.0 - max(f.x, max(f.y, f.z)));
    }

    private float transmission_layer_weight(const float3 wi, const float3 wo)
    {
        if (min(wi.z, -wo.z) < BSDF_MIN_COS_THETA)
            return 0.0;
        if (let state = transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u) {
                float3 h = thin_walled ? normalize(wi + float3(wo.x, wo.y, -wo.z))
                                       : (state.bulk.alpha == 0.0 ? float3(0.0, 0.0, 1.0)
                                                                  : microfacet_refraction::half_vector(wi, wo, eta));
                float wi_dot_h = saturate(dot(wi, h));
                float3 f = dielectric_fresnel(wi_dot_h);
                return max(0.0, 1.0 - max(f.x, max(f.y, f.z)));
            }
        }
        return 0.0;
    }

    private float3 anisotropic_local(float3 w, GLTFAnisotropyState state)
    {
        return float3(dot(w.xy, state.direction), dot(w.xy, float2(-state.direction.y, state.direction.x)), w.z);
    }

    private BSDFEval eval_anisotropic(float3 wi, float3 wo, GLTFAnisotropyState state)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return BSDFEval();
        wi = anisotropic_local(wi, state);
        wo = anisotropic_local(wo, state);
        float3 h = normalize(wi + wo);
        float wi_dot_h = dot(wi, h);
        return BSDFEval(
            float3(state.distribution.eval_ndf(h) * state.distribution.G2(wi, wo) / (4.0 * wi.z)),
            state.distribution.eval_pdf(wi, h) / (4.0 * wi_dot_h)
        );
    }

    private bool
    sample_anisotropic(float3 wi, GLTFAnisotropyState state, const BSDFSampleInput input, out BSDFSample sample)
    {
        sample = {};
        if (wi.z < BSDF_MIN_COS_THETA)
            return false;
        float3 local_wi = anisotropic_local(wi, state);
        float3 h = state.distribution.sample(local_wi, input.u_dir);
        float3 wo = 2.0 * dot(local_wi, h) * h - local_wi;
        sample.wo = float3(
            state.direction.x * wo.x - state.direction.y * wo.y,
            state.direction.y * wo.x + state.direction.x * wo.y,
            wo.z
        );
        sample.flags = BSDFFlags::glossy_reflection;
        return wo.z >= BSDF_MIN_COS_THETA;
    }

    public override float3 eval(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, bc).value;
    }

    public BSDFEval eval_with_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        BSDFEval result = {};
        if (p_diffuse > 0.0) {
            const BSDFEval lobe = diffuse.eval_with_pdf(wi, wo, bc);
            result.value += lobe.value * diffuse_layer_weight(wi, wo);
            result.pdf += p_diffuse * lobe.pdf;
        }
        if (p_specular > 0.0) {
            BSDFEval lobe = {};
            bool use_anisotropy = false;
            if (let state = anisotropy.get()) {
                if ((feature_mask & gltfmat::FeatureBits::ANISOTROPY) != 0u && state.active) {
                    lobe = eval_anisotropic(wi, wo, state);
                    use_anisotropy = true;
                }
            }
            if (!use_anisotropy)
                lobe = specular.eval_with_pdf(wi, wo, bc);
            result.value += specular_layer_weight(wi, wo) * lobe.value;
            result.pdf += p_specular * lobe.pdf;
        }
        if (let state = transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u && state.proposal_probability > 0.0) {
                const BSDFEval lobe
                    = thin_walled ? state.thin.eval_with_pdf(wi, wo, bc) : state.bulk.eval_with_pdf(wi, wo, bc);
                result.value += state.tint * transmission_layer_weight(wi, wo) * lobe.value;
                result.pdf += state.proposal_probability * lobe.pdf;
            }
        }
        if (let state = diffuse_transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u && state.proposal_probability > 0.0) {
                const float3 reflected_wo = float3(wo.xy, -wo.z);
                const BSDFEval lobe = state.lobe.eval_with_pdf(wi, reflected_wo, bc);
                result.value += lobe.value * diffuse_layer_weight(wi, reflected_wo);
                result.pdf += state.proposal_probability * lobe.pdf;
            }
        }
        if (let state = multiscatter.get()) {
            if ((feature_mask & gltfmat::FeatureBits::MULTISCATTER) != 0u && state.proposal_probability > 0.0) {
                const BSDFEval lobe = state.brdf.eval_with_pdf(wi, wo, bc);
                result.value += state.weight * lobe.value;
                result.pdf += state.proposal_probability * lobe.pdf;
            }
        }
        if (let state = sheen.get()) {
            if ((feature_mask & gltfmat::FeatureBits::SHEEN) != 0u && state.proposal_probability > 0.0) {
                result.value *= sheen_base_weight(wo);
                result.value += state.color * gltf_sheen::eval(wi, wo, state.roughness);
                if (min(wi.z, wo.z) >= BSDF_MIN_COS_THETA)
                    result.pdf += state.proposal_probability / (2.0 * float.PI);
            }
        }
        return result;
    }

    public bool sample(const float3 wi, const BSDFSampleInput input, const BSDFContext bc, out BSDFSample sample)
    {
        sample = {};
        float selection_pdf;

        float p_transmission = 0.0;
        if (let state = transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u)
                p_transmission = state.proposal_probability;
        }
        float p_multiscatter = 0.0;
        if (let state = multiscatter.get()) {
            if ((feature_mask & gltfmat::FeatureBits::MULTISCATTER) != 0u)
                p_multiscatter = state.proposal_probability;
        }
        float p_sheen = 0.0;
        if (let state = sheen.get()) {
            if ((feature_mask & gltfmat::FeatureBits::SHEEN) != 0u)
                p_sheen = state.proposal_probability;
        }
        float p_diffuse_transmission = 0.0;
        if (let state = diffuse_transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u)
                p_diffuse_transmission = state.proposal_probability;
        }

        // Select one proposal and sample it once. Rejected directions contribute zero; retrying
        // would change the distribution unless the PDF accounted for the conditional sampling.
        // None of the child samplers consume u_lobe, so the input does not need remapping.
        if (input.u_lobe < p_diffuse) {
            if (!diffuse.sample(wi, input, bc, sample))
                return false;
            selection_pdf = p_diffuse;
        } else if (input.u_lobe < p_diffuse + p_specular) {
            bool valid = false;
            bool use_anisotropy = false;
            if (let state = anisotropy.get()) {
                if ((feature_mask & gltfmat::FeatureBits::ANISOTROPY) != 0u && state.active) {
                    valid = sample_anisotropic(wi, state, input, sample);
                    use_anisotropy = true;
                }
            }
            if (!use_anisotropy)
                valid = specular.sample(wi, input, bc, sample);
            if (!valid)
                return false;
            selection_pdf = p_specular;
        } else if (input.u_lobe < p_diffuse + p_specular + p_transmission) {
            bool valid = false;
            if (let state = transmission.get()) {
                if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u)
                    valid = thin_walled ? state.thin.sample(wi, input, bc, sample)
                                        : state.bulk.sample(wi, input, bc, sample);
            }
            if (!valid)
                return false;
            selection_pdf = p_transmission;
        } else if (input.u_lobe < p_diffuse + p_specular + p_transmission + p_multiscatter && p_multiscatter > 0.0) {
            bool valid = false;
            if (let state = multiscatter.get()) {
                if ((feature_mask & gltfmat::FeatureBits::MULTISCATTER) != 0u)
                    valid = state.brdf.sample(wi, input, bc, sample);
            }
            if (!valid)
                return false;
            selection_pdf = p_multiscatter;
        } else if (input.u_lobe < p_diffuse + p_specular + p_transmission + p_multiscatter + p_sheen && p_sheen > 0.0) {
            // Uniform z and azimuth give uniform solid angle, not cosine-weighted sampling.
            float radius = sqrt(max(0.0, 1.0 - input.u_dir.x * input.u_dir.x));
            float phi = 2.0 * float.PI * input.u_dir.y;
            sample.wo = float3(radius * cos(phi), radius * sin(phi), input.u_dir.x);
            sample.flags = BSDFFlags::glossy_reflection;
            selection_pdf = p_sheen;
        } else if (p_diffuse_transmission > 0.0) {
            bool valid = false;
            if (let state = diffuse_transmission.get()) {
                if ((feature_mask & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u)
                    valid = state.lobe.sample(wi, input, bc, sample);
            }
            if (!valid)
                return false;
            sample.wo.z = -sample.wo.z;
            sample.flags = BSDFFlags::diffuse_transmission;
            selection_pdf = p_diffuse_transmission;
        } else {
            return false;
        }

        if (sample.has_flag(BSDFFlags::delta)) {
            float3 layer_weight;
            if (sample.has_flag(BSDFFlags::transmission)) {
                layer_weight = float3(0.0);
                if (let state = transmission.get()) {
                    if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u)
                        layer_weight = state.tint * transmission_layer_weight(wi, sample.wo);
                }
            } else {
                layer_weight = specular_layer_weight(wi, sample.wo);
            }
            sample.weight *= layer_weight * sheen_base_weight(sample.wo) / selection_pdf;
            return true;
        }

        // Continuous samples use the complete BSDF divided by the marginal mixture PDF.
        // This differs from the selected-component estimator used by Standard and SpecGloss.
        const BSDFEval evaluated = eval_with_pdf(wi, sample.wo, bc);
        if (evaluated.pdf <= 0.0)
            return false;
        sample.pdf = evaluated.pdf;
        sample.weight = evaluated.value / evaluated.pdf;
        return true;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        float3 rgb = dielectric_fresnel(wi.z);
        float f = max(rgb.x, max(rgb.y, rgb.z));
        float3 reflected = (thin_walled && iridescence_factor() <= 0.0 && !has_anisotropy())
            ? specular_weight * specular.eval_albedo(wi).reflection
            : reflection_fresnel(wi.z);
        float3 reflection = saturate((1.0 - f) * diffuse.albedo + reflected);
        if (let state = multiscatter.get()) {
            if ((feature_mask & gltfmat::FeatureBits::MULTISCATTER) != 0u && state.weight > 0.0)
                reflection = saturate(reflection + state.weight * state.brdf.eval_albedo(wi).reflection);
        }

        float3 transmitted = float3(0.0);
        float3 transmission_tint = float3(0.0);
        if (let state = transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u)
                transmission_tint = state.tint;
        }
        float3 diffuse_transmission_albedo = float3(0.0);
        if (let state = diffuse_transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u)
                diffuse_transmission_albedo = state.lobe.albedo;
        }
        transmitted = min((1.0 - f) * (transmission_tint + diffuse_transmission_albedo), 1.0 - reflection);
        if (let state = sheen.get()) {
            if ((feature_mask & gltfmat::FeatureBits::SHEEN) != 0u && state.proposal_probability > 0.0) {
                float scale = sheen_base_weight(wi);
                reflection = saturate(reflection * scale + state.color * state.incident_energy);
                transmitted = min(transmitted * scale, 1.0 - reflection);
            }
        }
        return AlbedoContributions(reflection, 1.0 - reflection - transmitted, transmitted, float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        if (let state = anisotropy.get()) {
            if ((feature_mask & gltfmat::FeatureBits::ANISOTROPY) != 0u && state.active)
                return RoughnessInformation(state.distribution.alpha, float3(0.0));
        }
        return RoughnessInformation(float2(specular.alpha), float3(0.0));
    }

    public float transmission_alpha()
    {
        if (let state = transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u)
                return thin_walled ? state.thin.alpha : state.bulk.alpha;
        }
        return 0.0;
    }

    public property float3 ior_as_reflectance
    {
        get {
            return (thin_walled && iridescence_factor() <= 0.0 && !has_anisotropy()) ? specular_weight * specular.albedo
                                                                                     : reflection_fresnel(1.0);
        }
    }
};
