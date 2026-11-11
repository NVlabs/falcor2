// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct GLTFVolumeState
{
    public float3 sigma_a = float3(0.0);
    public __init() { }
    public __init(float3 sigma_a) { this.sigma_a = sigma_a; }
};

public struct GLTFClearcoatState
{
    public Frame frame_ws;
    public float factor = 0.0;
    public float roughness = 0.0;
    public __init() { }
    public __init(Frame frame_ws, float factor, float roughness)
    {
        this.frame_ws = frame_ws;
        this.factor = factor;
        this.roughness = roughness;
    }
};

public struct GLTFMaterialInstance<let Features : uint> : IMaterialInstance
{
    public static const bool HAS_EMISSION = (Features & gltfmat::FeatureBits::EMISSION) != 0u;
    public static const bool HAS_VOLUME = (Features & gltfmat::FeatureBits::VOLUME) != 0u;
    public static const bool HAS_CLEARCOAT = (Features & gltfmat::FeatureBits::CLEARCOAT) != 0u;

    public uint feature_mask = 0u;
    public Frame shading_frame_ws;
    public GLTFBSDFData<Features> bsdf_data;
    public Conditional<float3, HAS_EMISSION> emission = Conditional<float3, HAS_EMISSION>(float3(0.0));
    public float interior_ior = 1.0;
    public Conditional<GLTFVolumeState, HAS_VOLUME> volume
        = Conditional<GLTFVolumeState, HAS_VOLUME>(GLTFVolumeState());
    public Conditional<GLTFClearcoatState, HAS_CLEARCOAT> clearcoat
        = Conditional<GLTFClearcoatState, HAS_CLEARCOAT>(GLTFClearcoatState());

    private float coat_weight(const SurfaceInteraction si, GLTFClearcoatState state)
    {
        // Khronos simple coat model uses view-normal Fresnel at IOR 1.5.
        return gltf_material::clearcoat_weight(state.factor, dot(si.wi_ws, state.frame_ws.normal));
    }

    private SpecularMicrofacetBRDF make_coat(GLTFClearcoatState state)
    {
        float alpha = state.roughness * state.roughness;
        return SpecularMicrofacetBRDF(float3(1.0), alpha < GLTF_MIN_GGX_ALPHA ? 0.0 : alpha, BSDFFlags::all);
    }

    public override float get_interior_ior() { return interior_ior; }

    public override HomogeneousMediumProperties get_interior_medium()
    {
        if (let state = volume.get()) {
            if ((feature_mask & gltfmat::FeatureBits::VOLUME) != 0u)
                return { state.sigma_a, float3(0.0), 0.0 };
        }
        return { float3(0.0), float3(0.0), 0.0 };
    }

    private GLTFBSDF<Features> make_bsdf(const SurfaceInteraction si)
    {
        return GLTFBSDF<Features>(shading_frame_ws.to_local(si.wi_ws), bsdf_data, feature_mask);
    }

    private BSDFEval eval_base(const SurfaceInteraction si, const float3 wo_ws)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);
        if (!surface_material::is_valid_hemisphere_reflection_or_transmission(
                si,
                shading_frame_ws,
                wi_ls,
                wo_ls,
                wo_ws
            ))
            return BSDFEval();
        GLTFBSDF<Features> bsdf = GLTFBSDF<Features>(wi_ls, bsdf_data, feature_mask);
        return bsdf.eval_with_pdf(wi_ls, wo_ls, BSDFContext());
    }

    private bool sample_base(const SurfaceInteraction si, const MaterialSampleInput input, out MaterialSample sample)
    {
        sample = {};
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        GLTFBSDF<Features> bsdf = GLTFBSDF<Features>(wi_ls, bsdf_data, feature_mask);
        BSDFSample bsdf_sample;
        if (!bsdf.sample(wi_ls, input, BSDFContext(), bsdf_sample))
            return false;
        sample.pdf = bsdf_sample.pdf;
        sample.weight = bsdf_sample.weight;
        sample.flags = bsdf_sample.flags;
        sample.wo_ws = shading_frame_ws.to_global(bsdf_sample.wo);
        return surface_material::is_valid_hemisphere_reflection_or_transmission(
            si,
            shading_frame_ws,
            wi_ls,
            bsdf_sample.wo,
            sample.wo_ws
        );
    }

    private bool valid_coat_direction(const SurfaceInteraction si, float3 wo_ws, GLTFClearcoatState state)
    {
        const float3 wi = state.frame_ws.to_local(si.wi_ws);
        const float3 wo = state.frame_ws.to_local(wo_ws);
        return wo.z > 0.0
            && surface_material::is_valid_hemisphere_reflection_or_transmission(si, state.frame_ws, wi, wo, wo_ws);
    }

    public BSDFEval
    eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both>(const SurfaceInteraction si, const float3 wo_ws)
    {
        BSDFEval result = eval_base(si, wo_ws);
        if (let state = clearcoat.get()) {
            if ((feature_mask & gltfmat::FeatureBits::CLEARCOAT) != 0u && state.factor > 0.0) {
                const float k = coat_weight(si, state);
                result.value *= 1.0 - k;
                result.pdf *= 1.0 - k;
                if (valid_coat_direction(si, wo_ws, state)) {
                    const BSDFEval coat = make_coat(state).eval_with_pdf(
                        state.frame_ws.to_local(si.wi_ws),
                        state.frame_ws.to_local(wo_ws),
                        BSDFContext()
                    );
                    result.value += k * coat.value;
                    result.pdf += k * coat.pdf;
                }
            }
        }
        return result;
    }

    public bool sample(const SurfaceInteraction si, const MaterialSampleInput input, out MaterialSample sample)
    {
        if (let state = clearcoat.get()) {
            if ((feature_mask & gltfmat::FeatureBits::CLEARCOAT) != 0u && state.factor > 0.0) {
                sample = {};
                const float k = coat_weight(si, state);
                MaterialSampleInput child_input = input;
                const bool sampled_coat = choose_sample_branch(child_input.u_lobe, k);
                if (sampled_coat) {
                    BSDFSample coat_sample;
                    if (!make_coat(state)
                             .sample(state.frame_ws.to_local(si.wi_ws), child_input, BSDFContext(), coat_sample))
                        return false;
                    sample.wo_ws = state.frame_ws.to_global(coat_sample.wo);
                    if (!valid_coat_direction(si, sample.wo_ws, state))
                        return false;
                    sample.weight = coat_sample.weight;
                    sample.pdf = k * coat_sample.pdf;
                    sample.flags = coat_sample.flags;
                } else if (!sample_base(si, child_input, sample)) {
                    return false;
                }
                if ((sample.flags & BSDFFlags::delta) != BSDFFlags::none)
                    return true;
                BSDFEval evaluated;
                if (sampled_coat) {
                    evaluated = eval_with_pdf(si, sample.wo_ws);
                } else {
                    // GLTFBSDF.sample() already evaluated the non-delta base
                    // mixture. Reuse it rather than preparing and evaluating
                    // the full base BSDF a second time.
                    evaluated.value = (1.0 - k) * (sample.weight * sample.pdf);
                    evaluated.pdf = (1.0 - k) * sample.pdf;
                    if (valid_coat_direction(si, sample.wo_ws, state)) {
                        const BSDFEval coat = make_coat(state).eval_with_pdf(
                            state.frame_ws.to_local(si.wi_ws),
                            state.frame_ws.to_local(sample.wo_ws),
                            BSDFContext()
                        );
                        evaluated.value += k * coat.value;
                        evaluated.pdf += k * coat.pdf;
                    }
                }
                if (evaluated.pdf <= 0.0)
                    return false;
                sample.pdf = evaluated.pdf;
                sample.weight = evaluated.value / evaluated.pdf;
                return true;
            }
        }
        return sample_base(si, input, sample);
    }

    private MaterialProperties collect_base_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = {};
        float metallic = saturate(bsdf_data.metallic);
        float alpha = bsdf_data.roughness * bsdf_data.roughness;
        result.material_color = saturate(bsdf_data.base_color);
        result.emission = eval_emissive_radiance();
        result.metallic = metallic;
        result.roughness = alpha < GLTF_MIN_GGX_ALPHA ? 0.0 : bsdf_data.roughness;
        result.guide_normal = shading_frame_ws.normal;

        float transmission_factor = 0.0;
        if (let inputs = bsdf_data.transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::TRANSMISSION) != 0u)
                transmission_factor = inputs.factor;
        }
        float diffuse_transmission_factor = 0.0;
        float3 diffuse_transmission_color = float3(1.0);
        if (let inputs = bsdf_data.diffuse_transmission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::DIFFUSE_TRANSMISSION) != 0u) {
                diffuse_transmission_factor = inputs.factor;
                diffuse_transmission_color = inputs.color;
            }
        }
        float diffuse_weight = (1.0 - metallic) * (1.0 - transmission_factor);
        result.diffuse_reflection_albedo = bsdf_data.base_color * diffuse_weight * (1.0 - diffuse_transmission_factor);
        result.diffuse_transmission_albedo = diffuse_transmission_color * diffuse_weight * diffuse_transmission_factor;
        result.specular_transmission_albedo = bsdf_data.base_color * (1.0 - metallic) * transmission_factor;
        result.specular_reflection_albedo = lerp(float3(bsdf_data.dielectric_f0), bsdf_data.base_color, metallic);
        if (let inputs = bsdf_data.iridescence.get()) {
            if ((feature_mask & gltfmat::FeatureBits::IRIDESCENCE) != 0u && inputs.factor > 0.0)
                result.specular_reflection_albedo = make_bsdf(si).ior_as_reflectance;
        }
        result.specular_reflectance = result.specular_reflection_albedo;
        return result;
    }

    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = collect_base_properties(si);
        if (let inputs = bsdf_data.sheen.get()) {
            if ((feature_mask & gltfmat::FeatureBits::SHEEN) != 0u && any(inputs.color > 0.0)) {
                float energy = gltf_sheen::energy(dot(si.wi_ws, shading_frame_ws.normal), inputs.roughness);
                float scale = 1.0 - max(inputs.color.x, max(inputs.color.y, inputs.color.z)) * energy;
                result.diffuse_reflection_albedo *= scale;
                result.diffuse_transmission_albedo *= scale;
                result.specular_transmission_albedo *= scale;
                result.specular_reflection_albedo = result.specular_reflection_albedo * scale + inputs.color * energy;
            }
        }
        if (let state = clearcoat.get()) {
            if ((feature_mask & gltfmat::FeatureBits::CLEARCOAT) != 0u && state.factor > 0.0) {
                const float k = coat_weight(si, state);
                result.diffuse_reflection_albedo *= 1.0 - k;
                result.diffuse_transmission_albedo *= 1.0 - k;
                result.specular_transmission_albedo *= 1.0 - k;
                result.specular_reflection_albedo = lerp(result.specular_reflection_albedo, float3(1.0), k);
                result.specular_reflectance = lerp(result.specular_reflectance, float3(1.0), 0.04 * state.factor);
            }
        }
        return result;
    }

    public override Optional<ExtraBsdfProperties>
    collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        ExtraBsdfProperties result = {};
        MaterialProperties properties = collect_base_properties(si);
        result.bsdf_count = any(properties.specular_transmission_albedo > 0.0) ? 3 : 2;
        for (uint i = 0; i < result.bsdf_count; ++i) {
            result.bsdf_N[i] = si.shading_frame_ws.to_local(shading_frame_ws.normal);
            result.bsdf_T[i] = si.shading_frame_ws.to_local(shading_frame_ws.tangent);
            result.bsdf_B[i] = si.shading_frame_ws.to_local(shading_frame_ws.bitangent);
            result.bsdf_weight[i] = float3(1.0);
        }
        result.bsdf_albedo[0] = properties.diffuse_reflection_albedo;
        result.bsdf_roughness[0] = float2(1.0);
        result.bsdf_albedo[1] = properties.specular_reflection_albedo;
        result.bsdf_roughness[1] = float2(properties.roughness);

        if (let inputs = bsdf_data.anisotropy.get()) {
            if ((feature_mask & gltfmat::FeatureBits::ANISOTROPY) != 0u && inputs.strength > 0.0) {
                float2 direction = inputs.direction;
                result.bsdf_T[1] = si.shading_frame_ws.to_local(shading_frame_ws.to_global(float3(direction, 0.0)));
                result.bsdf_B[1]
                    = si.shading_frame_ws.to_local(shading_frame_ws.to_global(float3(-direction.y, direction.x, 0.0)));
                result.bsdf_roughness[1] = make_bsdf(si).eval_roughness(shading_frame_ws.to_local(si.wi_ws)).roughness;
            }
        }
        if (result.bsdf_count == 3) {
            result.bsdf_albedo[2] = properties.specular_transmission_albedo;
            result.bsdf_roughness[2] = float2(make_bsdf(si).transmission_alpha());
        }
        if (any(properties.diffuse_transmission_albedo > 0.0)) {
            uint i = result.bsdf_count++;
            result.bsdf_N[i] = -result.bsdf_N[0];
            result.bsdf_T[i] = result.bsdf_T[0];
            result.bsdf_B[i] = -result.bsdf_B[0];
            result.bsdf_weight[i] = float3(1.0);
            result.bsdf_albedo[i] = properties.diffuse_transmission_albedo;
            result.bsdf_roughness[i] = float2(1.0);
        }

        if (let inputs = bsdf_data.sheen.get()) {
            if ((feature_mask & gltfmat::FeatureBits::SHEEN) != 0u && any(inputs.color > 0.0)) {
                float energy = max(
                    gltf_sheen::energy(dot(si.wi_ws, shading_frame_ws.normal), inputs.roughness),
                    gltf_sheen::energy(dot(wo, shading_frame_ws.normal), inputs.roughness)
                );
                float scale = 1.0 - max(inputs.color.x, max(inputs.color.y, inputs.color.z)) * energy;
                for (uint j = 0; j < result.bsdf_count; ++j)
                    result.bsdf_weight[j] *= scale;
                uint i = result.bsdf_count++;
                result.bsdf_N[i] = result.bsdf_N[0];
                result.bsdf_T[i] = result.bsdf_T[0];
                result.bsdf_B[i] = result.bsdf_B[0];
                result.bsdf_weight[i] = float3(1.0);
                result.bsdf_albedo[i] = inputs.color;
                result.bsdf_roughness[i] = float2(inputs.roughness * inputs.roughness);
            }
        }
        if (let state = clearcoat.get()) {
            if ((feature_mask & gltfmat::FeatureBits::CLEARCOAT) != 0u && state.factor > 0.0) {
                const float k = coat_weight(si, state);
                for (uint j = 0; j < result.bsdf_count; ++j)
                    result.bsdf_weight[j] *= 1.0 - k;
                uint i = result.bsdf_count++;
                result.bsdf_N[i] = si.shading_frame_ws.to_local(state.frame_ws.normal);
                result.bsdf_T[i] = si.shading_frame_ws.to_local(state.frame_ws.tangent);
                result.bsdf_B[i] = si.shading_frame_ws.to_local(state.frame_ws.bitangent);
                result.bsdf_weight[i] = float3(k);
                result.bsdf_albedo[i] = float3(1.0);
                result.bsdf_roughness[i] = float2(make_coat(state).alpha);
            }
        }
        return result;
    }

    public BSDFFlags get_lobe_types(const SurfaceInteraction si)
    {
        BSDFFlags result;
        // Core materials only need roughness and diffuse albedo to report their
        // lobes. Avoid preparing a full BSDF (and its multiscatter LUT lookup)
        // solely for direct-light hemisphere selection.
        if ((Features
             & (gltfmat::FeatureBits::ANISOTROPY | gltfmat::FeatureBits::TRANSMISSION
                | gltfmat::FeatureBits::DIFFUSE_TRANSMISSION | gltfmat::FeatureBits::SHEEN))
            == 0u) {
            const float metallic = saturate(bsdf_data.metallic);
            const float roughness = saturate(bsdf_data.roughness);
            result = roughness * roughness < GLTF_MIN_GGX_ALPHA ? BSDFFlags::delta_reflection
                                                                : BSDFFlags::glossy_reflection;
            if (any(bsdf_data.base_color * (1.0 - metallic) > 0.0))
                result |= BSDFFlags::diffuse_reflection;
        } else {
            result = make_bsdf(si).get_lobe_types();
        }
        if (let state = clearcoat.get()) {
            if ((feature_mask & gltfmat::FeatureBits::CLEARCOAT) != 0u && state.factor > 0.0)
                result |= make_coat(state).alpha == 0.0 ? BSDFFlags::delta_reflection : BSDFFlags::glossy_reflection;
        }
        return result;
    }

    public override bool is_emissive() { return any(eval_emissive_radiance() > 0.0); }

    public override float3 eval_emissive_radiance()
    {
        if (let value = emission.get()) {
            if ((feature_mask & gltfmat::FeatureBits::EMISSION) != 0u)
                return value;
        }
        return float3(0.0);
    }
};
