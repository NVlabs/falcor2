// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct GLTFMaterialInstance : IMaterialInstance
{
    public Frame shading_frame_ws;
    public GLTFBSDFData bsdf_data;
    public float3 emissive_radiance = {};
    public float interior_ior = 1.0;
    public float3 volume_sigma_a = {};
    public Frame clearcoat_frame_ws;
    public float clearcoat = 0.0;
    public float clearcoat_roughness = 0.0;

    public float coat_weight(const SurfaceInteraction si)
    {
        // Khronos' simple coat model uses the view-normal Fresnel term at IOR 1.5:
        // k = clearcoat * (0.04 + 0.96 * (1 - abs(Nc.V))^5).
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_clearcoat/README.md#layering
        return gltf_material::clearcoat_weight(clearcoat, dot(si.wi_ws, clearcoat_frame_ws.normal));
    }

    private SpecularMicrofacetBRDF make_coat()
    {
        // Fresnel is already in coat_weight(). Use a unit-Fresnel GGX lobe here
        // so it is applied once, with the coat's own squared roughness.
        float alpha = clearcoat_roughness * clearcoat_roughness;
        return SpecularMicrofacetBRDF(float3(1.0), alpha < GLTF_MIN_GGX_ALPHA ? 0.0 : alpha, BSDFFlags::all);
    }

    // Volume attenuation is absorption only; the glTF volume has no scattering term.
    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_volume/README.md
    public override float get_interior_ior() { return interior_ior; }
    public override HomogeneousMediumProperties get_interior_medium() { return { volume_sigma_a, float3(0.0), 0.0 }; }

    private GLTFBSDF make_bsdf(const SurfaceInteraction si)
    {
        return GLTFBSDF(shading_frame_ws.to_local(si.wi_ws), bsdf_data);
    }

    private BSDFEval eval_base<S : ISampleGenerator>(const SurfaceInteraction si, const float3 wo_ws, inout S sg)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);

        // Classify reflection/transmission in the normal-mapped frame, then check
        // that the directions agree with the geometric surface's sides.
        if (!surface_material::is_valid_hemisphere_reflection_or_transmission(
                si,
                shading_frame_ws,
                wi_ls,
                wo_ls,
                wo_ws
            ))
            return BSDFEval();
        GLTFBSDF bsdf = GLTFBSDF(wi_ls, bsdf_data);
        return bsdf.eval_with_pdf(wi_ls, wo_ls, sg, BSDFContext());
    }

    private bool sample_base<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out MaterialSample sample)
    {
        sample = {};
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        GLTFBSDF bsdf = GLTFBSDF(wi_ls, bsdf_data);
        BSDFSample bsdf_sample;
        if (!bsdf.sample(wi_ls, bsdf_sample, sg, BSDFContext()))
            return false;

        // An orthonormal frame change preserves solid angle and the sample weight.
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

    private bool valid_coat_direction(const SurfaceInteraction si, float3 wo_ws)
    {
        // The coat only reflects, and its normal map can differ from the base's.
        const float3 wi = clearcoat_frame_ws.to_local(si.wi_ws);
        const float3 wo = clearcoat_frame_ws.to_local(wo_ws);
        return wo.z > 0.0
            && surface_material::is_valid_hemisphere_reflection_or_transmission(si, clearcoat_frame_ws, wi, wo, wo_ws);
    }

    public BSDFEval eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both, S : ISampleGenerator>(
        const SurfaceInteraction si,
        const float3 wo_ws,
        inout S sg
    )
    {
        BSDFEval result = eval_base(si, wo_ws, sg);
        if (clearcoat <= 0.0)
            return result;

        // Layer value and sampling density use the same mixture coefficients.
        // Each lobe supplies its own outgoing cosine in its own shading frame.
        const float k = coat_weight(si);
        result.value *= 1.0 - k;
        result.pdf *= 1.0 - k;
        if (valid_coat_direction(si, wo_ws)) {
            const BSDFEval coat = make_coat().eval_with_pdf(
                clearcoat_frame_ws.to_local(si.wi_ws),
                clearcoat_frame_ws.to_local(wo_ws),
                sg,
                BSDFContext()
            );
            result.value += k * coat.value;
            result.pdf += k * coat.pdf;
        }
        return result;
    }

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out MaterialSample sample)
    {
        if (clearcoat <= 0.0)
            return sample_base(si, sg, sample);
        sample = {};
        const float k = coat_weight(si);

        // Use the layer's view-dependent weight as the initial proposal probability.
        // Physical attenuation and proposal division cancel for selected delta events.
        if (next_float(sg) < k) {
            BSDFSample coat_sample;
            if (!make_coat().sample(clearcoat_frame_ws.to_local(si.wi_ws), coat_sample, sg, BSDFContext()))
                return false;
            sample.wo_ws = clearcoat_frame_ws.to_global(coat_sample.wo);
            if (!valid_coat_direction(si, sample.wo_ws))
                return false;
            sample.weight = coat_sample.weight;
            sample.pdf = k * coat_sample.pdf;
            sample.flags = coat_sample.flags;
        } else if (!sample_base(si, sg, sample)) {
            return false;
        }

        // The renderer represents delta densities as zero. Keep a selected-component
        // estimator even if smooth base and coat happen to reflect into the same direction.
        if ((sample.flags & BSDFFlags::delta) != BSDFFlags::none)
            return true;

        // For continuous events, evaluate both layers at the chosen direction:
        // weight = ((1-k) * base + k * coat) / ((1-k) * base_pdf + k * coat_pdf).
        const BSDFEval evaluated = eval_with_pdf(si, sample.wo_ws, sg);
        if (evaluated.pdf <= 0.0)
            return false;
        sample.pdf = evaluated.pdf;
        sample.weight = evaluated.value / evaluated.pdf;
        return true;
    }

    private MaterialProperties collect_base_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = {};

        // These are inexpensive renderer guides, not numerical BSDF integrals.
        // Roughness is perceptual here; alpha is the squared microfacet width.
        float metallic = saturate(bsdf_data.metallic);
        float alpha = bsdf_data.roughness * bsdf_data.roughness;
        result.material_color = saturate(bsdf_data.base_color);
        result.emission = emissive_radiance;
        result.metallic = metallic;

        // TODO: Find a scalar roughness guide that accounts for anisotropy,
        // including glossy anisotropic lobes below the isotropic delta threshold.
        result.roughness = alpha < GLTF_MIN_GGX_ALPHA ? 0.0 : bsdf_data.roughness;
        result.guide_normal = shading_frame_ws.normal;

        // Metallic removes diffuse and transmission. Specular transmission removes
        // energy from both diffuse branches; diffuse transmission splits what remains.
        // These guide colors omit the BSDF's direction-dependent Fresnel attenuation.
        float diffuse_weight = (1.0 - metallic) * (1.0 - bsdf_data.transmission);
        result.diffuse_reflection_albedo
            = bsdf_data.base_color * diffuse_weight * (1.0 - bsdf_data.diffuse_transmission);
        result.diffuse_transmission_albedo
            = bsdf_data.diffuse_transmission_color * diffuse_weight * bsdf_data.diffuse_transmission;
        result.specular_transmission_albedo = bsdf_data.base_color * (1.0 - metallic) * bsdf_data.transmission;

        result.specular_reflection_albedo = lerp(float3(bsdf_data.dielectric_f0), bsdf_data.base_color, metallic);
        if (bsdf_data.iridescence > 0.0)
            result.specular_reflection_albedo = make_bsdf(si).ior_as_reflectance;

        // Reuse effective normal-incidence reflectance as the specular albedo estimate.
        result.specular_reflectance = result.specular_reflection_albedo;
        return result;
    }

    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = collect_base_properties(si);

        // Approximate integrated sheen layering using incident directional energy.
        // Full BSDF evaluation also considers outgoing energy when attenuating the base.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_sheen/README.md#albedo-scaling-technique
        if (any(bsdf_data.sheen_color > 0.0)) {
            float energy = gltf_sheen::energy(dot(si.wi_ws, shading_frame_ws.normal), bsdf_data.sheen_roughness);
            float scale
                = 1.0 - max(bsdf_data.sheen_color.x, max(bsdf_data.sheen_color.y, bsdf_data.sheen_color.z)) * energy;
            result.diffuse_reflection_albedo *= scale;
            result.diffuse_transmission_albedo *= scale;
            result.specular_transmission_albedo *= scale;
            result.specular_reflection_albedo
                = result.specular_reflection_albedo * scale + bsdf_data.sheen_color * energy;
        }

        // Clearcoat is above all base lobes, including sheen. Unit coat albedo is
        // a guide approximation; rough single-scattering GGX can reflect less than one.
        if (clearcoat > 0.0) {
            const float k = coat_weight(si);
            result.diffuse_reflection_albedo *= 1.0 - k;
            result.diffuse_transmission_albedo *= 1.0 - k;
            result.specular_transmission_albedo *= 1.0 - k;
            result.specular_reflection_albedo = lerp(result.specular_reflection_albedo, float3(1.0), k);
            result.specular_reflectance = lerp(result.specular_reflectance, float3(1.0), 0.04 * clearcoat);
        }
        return result;
    }

    public override Optional<ExtraBsdfProperties>
    collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        ExtraBsdfProperties result = {};

        // Start with unlayered guides so sheen and coat attenuation are applied once.
        // All axes are expressed relative to the surface interaction's shading frame.
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

        // Rotate the specular tangent axes by the authored anisotropy direction.
        // This branch reports GGX widths; the isotropic branch above reports perceptual roughness.
        if (bsdf_data.anisotropy_strength > 0.0) {
            float2 direction = bsdf_data.anisotropy_direction;
            result.bsdf_T[1] = si.shading_frame_ws.to_local(shading_frame_ws.to_global(float3(direction, 0.0)));
            result.bsdf_B[1]
                = si.shading_frame_ws.to_local(shading_frame_ws.to_global(float3(-direction.y, direction.x, 0.0)));
            result.bsdf_roughness[1] = make_bsdf(si).eval_roughness(shading_frame_ws.to_local(si.wi_ws)).roughness;
        }

        // Transmission uses the thin-sheet or bulk-refraction width actually sampled.
        if (result.bsdf_count == 3) {
            result.bsdf_albedo[2] = properties.specular_transmission_albedo;
            GLTFBSDF bsdf = make_bsdf(si);
            result.bsdf_roughness[2] = float2(bsdf_data.thin_walled ? bsdf.transmission.alpha : bsdf.refraction.alpha);
        }

        // Reverse normal and bitangent together to keep a consistent frame for
        // the diffuse lobe on the opposite side of the surface.
        if (any(properties.diffuse_transmission_albedo > 0.0)) {
            uint i = result.bsdf_count++;
            result.bsdf_N[i] = -result.bsdf_N[0];
            result.bsdf_T[i] = result.bsdf_T[0];
            result.bsdf_B[i] = -result.bsdf_B[0];
            result.bsdf_weight[i] = float3(1.0);
            result.bsdf_albedo[i] = properties.diffuse_transmission_albedo;
            result.bsdf_roughness[i] = float2(1.0);
        }

        // Pair-dependent sheen attenuation is 1 - max(color) * max(E(wi), E(wo)),
        // equivalent to taking the smaller of the two remaining-energy factors.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_sheen/README.md#albedo-scaling-technique
        if (any(bsdf_data.sheen_color > 0.0)) {
            float energy = max(
                gltf_sheen::energy(dot(si.wi_ws, shading_frame_ws.normal), bsdf_data.sheen_roughness),
                gltf_sheen::energy(dot(wo, shading_frame_ws.normal), bsdf_data.sheen_roughness)
            );
            float scale
                = 1.0 - max(bsdf_data.sheen_color.x, max(bsdf_data.sheen_color.y, bsdf_data.sheen_color.z)) * energy;
            for (uint j = 0; j < result.bsdf_count; ++j)
                result.bsdf_weight[j] *= scale;
            uint i = result.bsdf_count++;
            result.bsdf_N[i] = result.bsdf_N[0];
            result.bsdf_T[i] = result.bsdf_T[0];
            result.bsdf_B[i] = result.bsdf_B[0];
            result.bsdf_weight[i] = float3(1.0);
            result.bsdf_albedo[i] = bsdf_data.sheen_color;
            result.bsdf_roughness[i] = float2(bsdf_data.sheen_roughness * bsdf_data.sheen_roughness);
        }

        // Attenuate every underlying entry, then append the coat in its own frame.
        if (clearcoat > 0.0) {
            const float k = coat_weight(si);
            for (uint j = 0; j < result.bsdf_count; ++j)
                result.bsdf_weight[j] *= 1.0 - k;
            uint i = result.bsdf_count++;
            result.bsdf_N[i] = si.shading_frame_ws.to_local(clearcoat_frame_ws.normal);
            result.bsdf_T[i] = si.shading_frame_ws.to_local(clearcoat_frame_ws.tangent);
            result.bsdf_B[i] = si.shading_frame_ws.to_local(clearcoat_frame_ws.bitangent);
            result.bsdf_weight[i] = float3(k);
            result.bsdf_albedo[i] = float3(1.0);
            result.bsdf_roughness[i] = float2(make_coat().alpha);
        }
        return result;
    }

    public BSDFFlags get_lobe_types(const SurfaceInteraction si)
    {
        BSDFFlags result = make_bsdf(si).get_lobe_types();
        if (clearcoat > 0.0)
            result |= make_coat().alpha == 0.0 ? BSDFFlags::delta_reflection : BSDFFlags::glossy_reflection;
        return result;
    }

    public override bool is_emissive() { return any(emissive_radiance > 0.0); }

    public override float3 eval_emissive_radiance() { return emissive_radiance; }
};
