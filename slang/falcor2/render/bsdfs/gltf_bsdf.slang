// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

// Preserve Standard's smooth-lobe approximation during extraction. This is not
// a glTF requirement: alpha below this value is represented by a delta event.
public static const float GLTF_MIN_GGX_ALPHA = 0.0064;

public struct GLTFBSDFData : IDifferentiable
{
    // Resolved linear parameters; GLTFMaterial handles texture lookup and color decoding.
    public float3 base_color = {};
    public float metallic = {};
    public float roughness = {};
    public float anisotropy_strength = {};
    public float2 anisotropy_direction = float2(1.0, 0.0);
    public float diffuse_transmission = {};
    public float3 diffuse_transmission_color = float3(1.0);
    public float3 sheen_color = {};
    public float sheen_roughness = {};
    public float iridescence = {};
    public float iridescence_ior = 1.3;
    public float iridescence_thickness = 400.0;

    // Dielectric endpoints already include the specular extension's tint and strength.
    public float3 dielectric_f0 = float3(0.04);
    public float dielectric_f90 = 1.0;
    public float transmission = {};
    public float ior = 1.5;

    // Incident/transmitted IOR ratio for bulk refraction; distinct from the material's absolute IOR.
    public float eta = 1.0;
    public no_diff bool thin_walled = true;
    /// Prepared scene materials enable this when the shared scattering table is bound.
    public no_diff bool compensate_metal = false;
};

/// glTF metallic-roughness composition with thin-sheet or refractive transmission.
/// Values include the outgoing cosine. Clearcoat is composed separately by GLTFMaterialInstance.
/// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#material-structure
public struct GLTFBSDF : IBSDF, IDifferentiable
{
    public LambertDiffuseBRDF diffuse = {};
    public LambertDiffuseBRDF diffuse_transmission = {};
    public float p_diffuse_transmission = {};
    public float2 anisotropy_direction = float2(1.0, 0.0);
    public AnisotropicGGXPolicy<10000, true> anisotropic_distribution = {};
    public no_diff bool anisotropic = false;
    public SpecularMicrofacetBRDF specular = {};
    public ThinSpecularMicrofacetBTDF transmission = {};
    public RefractiveSpecularMicrofacetBTDF refraction = {};
    public no_diff bool thin_walled = true;
    public float metallic = {};
    public float3 metal_f0 = {};
    public float3 transmission_tint = {};
    public GGXMultipleScatteringBRDF multiscatter = {};
    public float multiscatter_weight = {};
    public float specular_weight = 1.0;
    public float3 sheen_color = {};
    public float sheen_roughness = {};
    public float iridescence = {};
    public float iridescence_ior = 1.3;
    public float iridescence_thickness = 400.0;
    public float3 dielectric_f0 = float3(0.04);
    public float dielectric_f90 = 1.0;
    public float p_sheen = {};
    public float sheen_incident_energy = {};
    public float p_diffuse = {};
    public float p_specular = {};
    public float p_transmission = {};
    public float p_multiscatter = {};
    public BSDFFlags lobe_types = BSDFFlags::none;

    public __init() { }

    public __init(const float3 wi, const GLTFBSDFData data)
    {
        // Preparation depends on wi: sampling probabilities and table energies belong to this incident direction.
        float metallic = saturate(data.metallic);
        this.metallic = metallic;
        metal_f0 = data.base_color;
        float anisotropy = saturate(data.anisotropy_strength);
        anisotropic = anisotropy > 0.0;
        // The material supplies a normalized tangent-plane direction.
        anisotropy_direction = data.anisotropy_direction;
        thin_walled = data.thin_walled;
        sheen_color = saturate(data.sheen_color);
        sheen_roughness = saturate(data.sheen_roughness);
        iridescence = data.iridescence_thickness > 0.0 ? saturate(data.iridescence) : 0.0;
        iridescence_ior = data.iridescence_ior;
        iridescence_thickness = data.iridescence_thickness;

        // The table integrates unit-color sheen. The floor keeps its sampling proposal supported;
        // it does not add physical sheen energy.
        if (any(sheen_color > 0.0)) {
            sheen_incident_energy = gltf_sheen::energy(wi.z, sheen_roughness);
            p_sheen = max(0.01, luminance(sheen_color) * sheen_incident_energy);
        }

        // glTF broadens the tangent GGX width by anisotropy^2, leaving the perpendicular width at r^2.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_anisotropy/README.md
        float roughness = saturate(data.roughness);
        dielectric_f0 = saturate(data.dielectric_f0);
        dielectric_f90 = saturate(data.dielectric_f90);
        float alpha = roughness * roughness;
        anisotropic_distribution
            = AnisotropicGGXPolicy<10000, true>(float2(lerp(alpha, 1.0, anisotropy * anisotropy), alpha));

        // Anisotropic reflection stays continuous, with widths clamped to 1e-4 by its policy.
        // Isotropic reflection and bulk refraction use the existing smooth-lobe approximation.
        bool is_delta = !anisotropic && alpha < GLTF_MIN_GGX_ALPHA;
        if (alpha < GLTF_MIN_GGX_ALPHA)
            alpha = 0.0;

        // Nested mixes give weights (1-m)(1-T)(1-D), (1-m)(1-T)D and (1-m)T.
        // Fresnel is direction-dependent and is applied later, separately from these colors.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_diffuse_transmission/README.md#khr_materials_transmission
        float T = saturate(data.transmission);
        float D = saturate(data.diffuse_transmission);
        diffuse = LambertDiffuseBRDF(data.base_color * (1.0 - metallic) * (1.0 - T) * (1.0 - D));
        diffuse_transmission = LambertDiffuseBRDF(data.diffuse_transmission_color * (1.0 - metallic) * (1.0 - T) * D);
        transmission_tint = data.base_color * (1.0 - metallic) * T;

        // Thin microfacets cease to refract when the material matches the surrounding air.
        // IOR zero is glTF's infinite-IOR compatibility sentinel.
        // This heuristic scales perceptual roughness before squaring, giving alpha_T = r^2 * scale^2.
        // Khronos' sample renderer instead scales alpha directly in its punctual-light path:
        // https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/main/source/Renderer/shaders/punctual.glsl
        float transmission_roughness = roughness * (data.ior == 0.0 ? 1.0 : saturate(2.0 * data.ior - 2.0));
        // Apply the smooth-lobe approximation after IOR narrowing, independently of reflection anisotropy.
        float transmission_alpha = transmission_roughness * transmission_roughness;
        if (transmission_alpha < GLTF_MIN_GGX_ALPHA)
            transmission_alpha = 0.0;
        transmission = ThinSpecularMicrofacetBTDF(transmission_alpha);
        refraction = RefractiveSpecularMicrofacetBTDF(alpha, data.eta);

        // Compensate only the ordinary metal share. This is an additional
        // multiple-scattering approximation, not another authored glTF layer.
        if (data.compensate_metal && !is_delta && metallic > 0.0 && iridescence < 1.0) {
            // Enterprise PBR approximates anisotropic compensation using r_uv = sqrt(r_u * r_v).
            // Our stored, clamped widths are alpha = r^2; the table takes perceptual roughness.
            // https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2021x.md.html#energycompensation
            float2 anisotropic_alpha = anisotropic_distribution.alpha;
            float compensation_roughness
                = anisotropic ? sqrt(sqrt(anisotropic_alpha.x * anisotropic_alpha.y)) : roughness;
            multiscatter = GGXMultipleScatteringBRDF(wi, compensation_roughness, data.base_color);
            // The Schlick table does not describe thin-film interference.
            // Keep compensation on the ordinary conductor portion of the glTF blend.
            multiscatter_weight = metallic * (1.0 - iridescence);
            p_multiscatter = multiscatter_weight * luminance(multiscatter.eval_albedo(wi).reflection);
        }

        // Mix dielectric and metal Fresnel endpoints before evaluating the shared GGX shape.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md#implementation
        specular_weight = lerp(dielectric_f90, 1.0, metallic);
        float3 specular_f0 = lerp(dielectric_f0, data.base_color, metallic);

        // Schlick is linear in F0 and F90. Factor out F90 so the existing
        // unit-grazing GGX leaf can be shared without changing its data contract.
        // F0 + (F90-F0)t = F90 * [F0/F90 + (1-F0/F90)t], where t = (1-cosine)^5.
        specular = SpecularMicrofacetBRDF(
            specular_weight > 0.0 ? specular_f0 / specular_weight : float3(0.0),
            alpha,
            BSDFFlags::all
        );

        if (!thin_walled || iridescence > 0.0 || anisotropic) {
            // Reuse the unit reflection shape, applying interface Fresnel below.
            specular.albedo = float3(1.0);
            specular_weight = 1.0;
        }

        // Describe the supported event types independently of their selection probabilities.
        lobe_types = is_delta ? BSDFFlags::delta_reflection : BSDFFlags::glossy_reflection;
        if (p_sheen > 0.0)
            lobe_types |= BSDFFlags::glossy_reflection;
        if (any(diffuse.albedo > 0.0))
            lobe_types = BSDFFlags(uint(lobe_types) | uint(BSDFFlags::diffuse_reflection));
        if (any(diffuse_transmission.albedo > 0.0))
            lobe_types |= BSDFFlags::diffuse_transmission;
        if (any(transmission_tint > 0.0))
            lobe_types = BSDFFlags(
                uint(lobe_types)
                | uint(
                    (thin_walled ? transmission.alpha : refraction.alpha) == 0.0 ? BSDFFlags::delta_transmission
                                                                                 : BSDFFlags::glossy_transmission
                )
            );

        // Retain the baseline proposal and random sequence, independently of the
        // physical coefficients above. In particular, metallic also biases the
        // diffuse selection probability; it must not attenuate eval a second time.
        p_diffuse = luminance(diffuse.albedo) * (1.0 - metallic);
        p_diffuse_transmission = luminance(diffuse_transmission.albedo);
        p_specular = specular_weight * luminance(eval_fresnel_schlick(specular.albedo, float3(1.0), wi.z));
        if (!thin_walled || iridescence > 0.0 || anisotropic)
            p_specular = max(0.01, luminance(reflection_fresnel(wi.z)));

        // Black F0 still reflects oblique half-vectors. Keep its proposal supported
        // even when the view-normal Fresnel estimate vanishes at normal incidence.
        if (p_specular == 0.0 && specular_weight > 0.0)
            p_specular = 0.01 * specular_weight;
        p_transmission = luminance(transmission_tint);

        // Bias sampling towards sheen when it covers the base, retaining support for every base lobe.
        if (p_sheen > 0.0) {
            float base_scale
                = max(0.01, 1.0 - max(sheen_color.x, max(sheen_color.y, sheen_color.z)) * sheen_incident_energy);
            p_diffuse *= base_scale;
            p_diffuse_transmission *= base_scale;
            p_specular *= base_scale;
            p_transmission *= base_scale;
            p_multiscatter *= base_scale;
        }

        // Only these proposal weights are normalized. Physical BSDF coefficients remain unchanged.
        float norm = p_diffuse + p_specular + p_transmission + p_multiscatter + p_sheen + p_diffuse_transmission;
        if (norm > 0.0) {
            norm = 1.0 / norm;
            p_sheen *= norm;
            p_diffuse *= norm;
            p_diffuse_transmission *= norm;
            p_specular *= norm;
            p_transmission *= norm;
            p_multiscatter *= norm;
        }
    }

    public BSDFFlags get_lobe_types() { return lobe_types; }

    [Differentiable]
    private float3 dielectric_fresnel(float cosine)
    {
        // Snell: cos(theta_t)^2 = 1 - eta^2 * (1 - cos(theta_i)^2).
        // Bulk interfaces fully reflect under TIR. On exit, use the lower-IOR-side
        // cosine in the Schlick approximation; this is not an exact Fresnel evaluation.
        if (!thin_walled && refraction.eta != 1.0) {
            float ct2 = microfacet_refraction::transmitted_cosine_squared(cosine, refraction.eta);
            if (ct2 <= 0.0)
                return float3(1.0);
            if (refraction.eta > 1.0)
                cosine = sqrt(ct2);
        }

        // At a bulk boundary, eta == 1 means both media have the same IOR, so there is no dielectric reflection.
        // Bypass Schlick: even with F0 == 0, its F90 endpoint can produce nonzero reflection at oblique angles.
        // Thin sheets set eta to 1 because they do not use the bulk refraction ratio; they must still use Fresnel.
        // This only removes the ordinary interface response. An iridescent film can still reflect below.
        float3 f = !thin_walled && refraction.eta == 1.0
            ? float3(0.0)
            : eval_fresnel_schlick(dielectric_f0, float3(dielectric_f90), cosine);

        // Interpolate ordinary and thin-film Fresnel. The film receives the prepared, weighted F0.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_iridescence/README.md#dielectric-base-material-1
        if (iridescence > 0.0)
            f = lerp(
                f,
                gltf_iridescence::fresnel(cosine, iridescence_ior, iridescence_thickness, dielectric_f0),
                iridescence
            );
        return f;
    }

    [Differentiable]
    private float3 reflection_fresnel(float cosine)
    {
        // Metallic blends the dielectric and conductor interface responses.
        float3 metal = eval_fresnel_schlick(metal_f0, float3(1.0), cosine);
        if (iridescence > 0.0)
            metal = lerp(
                metal,
                gltf_iridescence::fresnel(cosine, iridescence_ior, iridescence_thickness, metal_f0),
                iridescence
            );
        return lerp(dielectric_fresnel(cosine), metal, metallic);
    }

    [Differentiable]
    private float sheen_base_weight(float3 wo)
    {
        if (p_sheen <= 0.0)
            return 1.0;

        // Symmetric attenuation uses the larger directional sheen energy, preserving reciprocity.
        // The table lookup is treated as constant by automatic differentiation.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_sheen/README.md#albedo-scaling-technique
        float outgoing_energy = no_diff gltf_sheen::energy(wo.z, sheen_roughness);
        return max(
            0.0,
            1.0 - max(sheen_color.x, max(sheen_color.y, sheen_color.z)) * max(sheen_incident_energy, outgoing_energy)
        );
    }

    [Differentiable]
    private float3 specular_layer_weight(const float3 wi, const float3 wo)
    {
        // The ordinary leaf already contains normalized Schlick Fresnel. Other paths
        // use a unit-Fresnel shape and supply the complete response at the reflection half-vector.
        return (thin_walled && iridescence <= 0.0 && !anisotropic) || wo.z <= 0.0
            ? float3(specular_weight)
            : reflection_fresnel(saturate(dot(wi, normalize(wi + wo))));
    }

    [Differentiable]
    private float diffuse_layer_weight(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return 0.0;

        // Reserve the largest reflected color channel, preserving the underlying diffuse hue.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md#implementation
        float wi_dot_h = saturate(dot(wi, normalize(wi + wo)));
        float3 f = dielectric_fresnel(wi_dot_h);
        return max(0.0, 1.0 - max(f.x, max(f.y, f.z)));
    }

    [Differentiable]
    private float transmission_layer_weight(const float3 wi, const float3 wo)
    {
        if (min(wi.z, -wo.z) < BSDF_MIN_COS_THETA)
            return 0.0;

        // Thin transmission reflects wo across the tangent plane to form its half-vector.
        // Bulk transmission uses Snell's half-vector, or the surface normal for a delta event.
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md#implementation-notes
        float3 h = thin_walled ? normalize(wi + float3(wo.x, wo.y, -wo.z))
                               : (refraction.alpha == 0.0 ? float3(0.0, 0.0, 1.0)
                                                          : microfacet_refraction::half_vector(wi, wo, refraction.eta));
        float wi_dot_h = saturate(dot(wi, h));
        float3 f = dielectric_fresnel(wi_dot_h);
        return max(0.0, 1.0 - max(f.x, max(f.y, f.z)));
    }

    [Differentiable]
    private float3 anisotropic_local(float3 w)
    {
        // Rotate into the anisotropy axes (d.x,d.y) and (-d.y,d.x), preserving z and solid angle.
        return float3(
            dot(w.xy, anisotropy_direction),
            dot(w.xy, float2(-anisotropy_direction.y, anisotropy_direction.x)),
            w.z
        );
    }

    [Differentiable]
    private BSDFEval eval_anisotropic(float3 wi, float3 wo)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return BSDFEval();
        wi = anisotropic_local(wi);
        wo = anisotropic_local(wo);
        float3 h = normalize(wi + wo);
        float wi_dot_h = dot(wi, h);

        // The outgoing cosine cancels from D*G/(4*cos_i*cos_o). Fresnel is supplied by the caller.
        // Convert the sampled half-vector density to outgoing-direction density with 1/(4*wi.h).
        return BSDFEval(
            float3(anisotropic_distribution.eval_ndf(h) * anisotropic_distribution.G2(wi, wo) / (4.0 * wi.z)),
            anisotropic_distribution.eval_pdf(wi, h) / (4.0 * wi_dot_h)
        );
    }

    private bool sample_anisotropic<S : ISampleGenerator>(float3 wi, out BSDFSample sample, inout S sg)
    {
        sample = {};
        if (wi.z < BSDF_MIN_COS_THETA)
            return false;
        float3 local_wi = anisotropic_local(wi);
        float3 h = anisotropic_distribution.sample(local_wi, sg);
        float3 wo = 2.0 * dot(local_wi, h) * h - local_wi;

        // Undo the tangent-plane rotation. The enclosing sampler evaluates the complete mixture
        // to obtain weight and PDF; this proposal only supplies a direction and event type.
        sample.wo = float3(
            anisotropy_direction.x * wo.x - anisotropy_direction.y * wo.y,
            anisotropy_direction.y * wo.x + anisotropy_direction.x * wo.y,
            wo.z
        );
        sample.flags = BSDFFlags::glossy_reflection;
        return wo.z >= BSDF_MIN_COS_THETA;
    }

    [Differentiable]
    public override
        float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        return eval_with_pdf(wi, wo, sg, bc).value;
    }

    [Differentiable]
    public BSDFEval
    eval_with_pdf<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        BSDFEval result = {};

        // Physical values add with their material coefficients. PDFs mix with proposal probabilities.
        // All leaves include the outgoing cosine; do not multiply by another cosine here.
        if (p_diffuse > 0.0) {
            const BSDFEval lobe = diffuse.eval_with_pdf(wi, wo, sg, bc);
            result.value += lobe.value * diffuse_layer_weight(wi, wo);
            result.pdf += p_diffuse * lobe.pdf;
        }
        if (p_specular > 0.0) {
            const BSDFEval lobe = anisotropic ? eval_anisotropic(wi, wo) : specular.eval_with_pdf(wi, wo, sg, bc);
            result.value += specular_layer_weight(wi, wo) * lobe.value;
            result.pdf += p_specular * lobe.pdf;
        }
        if (p_transmission > 0.0) {
            const BSDFEval lobe
                = thin_walled ? transmission.eval_with_pdf(wi, wo, sg, bc) : refraction.eval_with_pdf(wi, wo, sg, bc);
            result.value += transmission_tint * transmission_layer_weight(wi, wo) * lobe.value;
            result.pdf += p_transmission * lobe.pdf;
        }
        if (p_diffuse_transmission > 0.0) {
            // Mirror only z to reuse Lambert's upper-hemisphere lobe; the solid-angle Jacobian is one.
            const float3 reflected_wo = float3(wo.xy, -wo.z);
            const BSDFEval lobe = diffuse_transmission.eval_with_pdf(wi, reflected_wo, sg, bc);
            result.value += lobe.value * diffuse_layer_weight(wi, reflected_wo);
            result.pdf += p_diffuse_transmission * lobe.pdf;
        }
        if (p_multiscatter > 0.0) {
            const BSDFEval lobe = multiscatter.eval_with_pdf(wi, wo, sg, bc);
            result.value += multiscatter_weight * lobe.value;
            result.pdf += p_multiscatter * lobe.pdf;
        }
        if (p_sheen > 0.0) {
            result.value *= sheen_base_weight(wo);
            result.value += sheen_color * gltf_sheen::eval(wi, wo, sheen_roughness);

            // The sheen proposal is uniform over the upper hemisphere, whose solid angle is 2*pi.
            if (min(wi.z, wo.z) >= BSDF_MIN_COS_THETA)
                result.pdf += p_sheen / (2.0 * float.PI);
        }
        return result;
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg, const BSDFContext bc)
    {
        sample = {};
        float u_select = next_float(sg);
        float selection_pdf;

        // Select one proposal and sample it once. Rejected directions contribute zero; retrying
        // would change the distribution unless the PDF accounted for the conditional sampling.
        if (u_select < p_diffuse) {
            if (!diffuse.sample(wi, sample, sg, bc))
                return false;
            selection_pdf = p_diffuse;
        } else if (u_select < p_diffuse + p_specular) {
            if (!(anisotropic ? sample_anisotropic(wi, sample, sg) : specular.sample(wi, sample, sg, bc)))
                return false;
            selection_pdf = p_specular;
        } else if (u_select < p_diffuse + p_specular + p_transmission) {
            bool valid = thin_walled ? transmission.sample(wi, sample, sg, bc) : refraction.sample(wi, sample, sg, bc);
            if (!valid)
                return false;
            selection_pdf = p_transmission;
        } else if (u_select < p_diffuse + p_specular + p_transmission + p_multiscatter && p_multiscatter > 0.0) {
            if (!multiscatter.sample(wi, sample, sg, bc))
                return false;
            selection_pdf = p_multiscatter;
        } else if (u_select < p_diffuse + p_specular + p_transmission + p_multiscatter + p_sheen && p_sheen > 0.0) {
            // Uniform z and azimuth give uniform solid angle, not cosine-weighted sampling.
            float2 u = next_float2(sg);
            float radius = sqrt(max(0.0, 1.0 - u.x * u.x));
            float phi = 2.0 * float.PI * u.y;
            sample.wo = float3(radius * cos(phi), radius * sin(phi), u.x);
            sample.flags = BSDFFlags::glossy_reflection;
            selection_pdf = p_sheen;
        } else if (p_diffuse_transmission > 0.0) {
            if (!diffuse_transmission.sample(wi, sample, sg, bc))
                return false;
            sample.wo.z = -sample.wo.z;
            sample.flags = BSDFFlags::diffuse_transmission;
            selection_pdf = p_diffuse_transmission;
        } else {
            return false;
        }

        // Smooth leaves carry their integrated event weight and use a zero-PDF
        // sentinel. Their physical coefficients are shared with evaluation above.
        if (sample.has_flag(BSDFFlags::delta)) {
            float3 layer_weight = sample.has_flag(BSDFFlags::transmission)
                ? transmission_tint * transmission_layer_weight(wi, sample.wo)
                : specular_layer_weight(wi, sample.wo);
            sample.weight *= layer_weight * sheen_base_weight(sample.wo) / selection_pdf;
            return true;
        }

        // Continuous samples use the complete BSDF divided by the marginal mixture PDF.
        // This differs from the selected-component estimator used by Standard and SpecGloss.
        const BSDFEval evaluated = eval_with_pdf(wi, sample.wo, sg, bc);
        if (evaluated.pdf <= 0.0)
            return false;
        sample.pdf = evaluated.pdf;
        sample.weight = evaluated.value / evaluated.pdf;
        return true;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        // A bounded guide estimate, not integration of the half-vector coupling.
        float3 rgb = dielectric_fresnel(wi.z);
        float f = max(rgb.x, max(rgb.y, rgb.z));
        float3 reflected = (thin_walled && iridescence <= 0.0 && !anisotropic)
            ? specular_weight * specular.eval_albedo(wi).reflection
            : reflection_fresnel(wi.z);
        float3 reflection = saturate((1.0 - f) * diffuse.albedo + reflected);
        reflection = saturate(reflection + multiscatter_weight * multiscatter.eval_albedo(wi).reflection);

        // Bound the guide partition. These clamps do not alter eval() and cannot prove energy conservation.
        float3 transmitted = min((1.0 - f) * (transmission_tint + diffuse_transmission.albedo), 1.0 - reflection);
        if (p_sheen > 0.0) {
            float scale = sheen_base_weight(wi);
            reflection = saturate(reflection * scale + sheen_color * sheen_incident_energy);
            transmitted = min(transmitted * scale, 1.0 - reflection);
        }
        return AlbedoContributions(reflection, 1.0 - reflection - transmitted, transmitted, float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        // Report GGX widths, including the anisotropic policy's clamp, not perceptual roughness.
        return RoughnessInformation(anisotropic ? anisotropic_distribution.alpha : float2(specular.alpha), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            // Effective base-interface reflection at normal incidence; sheen and clearcoat are separate layers.
            return (thin_walled && iridescence <= 0.0 && !anisotropic) ? specular_weight * specular.albedo
                                                                       : reflection_fresnel(1.0);
        }
    }
};
