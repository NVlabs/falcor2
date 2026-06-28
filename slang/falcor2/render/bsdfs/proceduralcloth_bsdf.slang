// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Implements https://research.nvidia.com/labs/rtr/publication/weidlich2023hierarchical/

public struct ProceduralClothBSDF : IBSDF
{
    float2 uv_top = {};
    float2 uv_bottom = {};
    float yarn_id = {};
    float3 yarn_direction_top = {};
    float3 yarn_direction_bottom = {};
    float2 yarn_scale = {};
    float ply_twist = {};
    float ply_scale = {};
    float ply_count = {};
    float fiber_twist_angle = {};
    float fiber_reflectance = {};
    float fiber_transparency = {};
    float fiber_roughness_m = {};
    float fiber_roughness_n = {};
    float fiber_count = {};
    float3 fiber_color_top = {};
    float3 fiber_color_bottom = {};

    static const float SQRT_2 = sqrt(2.f);
    static const float C_UNIFORM_ROUGHNESS = 0.9069f;
    static const float C_MIN_ROUGHNESS = 0.0001f;
    static const float C_MAX_ROUGHNESS = C_UNIFORM_ROUGHNESS;
    static const float C_MIN_ANGLE = 0.0001f;

    const static int C_MAX_PLIES = 25;

    /// Classic rendering functions

    public __init() { }

    public __init(
        float2 uv_top,
        float2 uv_bottom,
        float yarn_id,
        float3 yarn_direction_top,
        float3 yarn_direction_bottom,
        float2 yarn_scale,
        float ply_twist,
        float ply_scale,
        float ply_count,
        float fiber_twist_angle,
        float fiber_reflectance,
        float fiber_transparency,
        float fiber_roughness_m,
        float fiber_roughness_n,
        float fiber_count,
        float3 fiber_color_top,
        float3 fiber_color_bottom
    )
    {
        this.uv_top = uv_top;
        this.uv_bottom = uv_bottom;
        this.yarn_id = yarn_id;
        this.yarn_direction_top = yarn_direction_top;
        this.yarn_direction_bottom = yarn_direction_bottom;
        this.yarn_scale = yarn_scale;
        this.ply_twist = ply_twist;
        this.ply_scale = ply_scale;
        this.ply_count = ply_count;
        this.fiber_twist_angle = fiber_twist_angle;
        this.fiber_reflectance = fiber_reflectance;
        this.fiber_transparency = fiber_transparency;
        this.fiber_roughness_m = fiber_roughness_m;
        this.fiber_roughness_n = fiber_roughness_n;
        this.fiber_count = fiber_count;
        this.fiber_color_top = fiber_color_top;
        this.fiber_color_bottom = fiber_color_bottom;
    }


    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, BSDFContext bc)
    {
        float2 closest_hit;
        bool hit_top;
        float3 roughness_yarn_fm2, roughness_yarn_fn2, roughness_yarn_bm2;
        float3 attenuation_f, attenuation_b;
        Intersection intersections[C_MAX_PLIES];
        int hit_counter;
        const int hit_id = eval_fiber_position(wi, false, closest_hit, hit_top, intersections, hit_counter);

        float3 contribution_eval = float3(0.f);

        if (hit_id > -1) {
            const float3 diffuse = eval_fiber_contribution(
                wi,
                closest_hit,
                hit_id,
                hit_top,
                intersections,
                hit_counter,
                roughness_yarn_fm2,
                roughness_yarn_fn2,
                attenuation_f,
                roughness_yarn_bm2,
                attenuation_b
            );

            Frame sf_yarn = get_yarn_frame(hit_top);
            Frame sf_ply, sf_fiber;
            const float z = get_z(hit_top, wi, sf_yarn.bitangent);
            get_frames(hit_id, z, closest_hit, sf_fiber, sf_ply);

            float3 wi_yarn, wo_yarn, wi_ply, wo_ply, wi_fiber, wo_fiber;
            float theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber, theta_iply, theta_oply, phi_iply, phi_oply,
                cos_wo_tfiber, cos_wo_tply;
            get_angles(
                wi,
                wo,
                sf_yarn,
                sf_ply,
                sf_fiber,
                wi_yarn,
                wo_yarn,
                wi_ply,
                wo_ply,
                wi_fiber,
                wo_fiber,
                theta_ifiber,
                theta_ofiber,
                phi_ifiber,
                phi_ofiber,
                theta_iply,
                theta_oply,
                phi_iply,
                phi_oply,
                cos_wo_tfiber,
                cos_wo_tply
            );

            const float2 uv = hit_top ? uv_top : uv_bottom;
            const float3 inter_fiber_schadowing = eval_local_shadowing(
                wo,
                closest_hit,
                get_h(hit_top),
                get_z(hit_top, wi, sf_yarn.bitangent),
                hit_id,
                hit_top
            );

            const float fresnel = eval_fresnel_schlick(fiber_reflectance, 1.f, cos(theta_ifiber));

            // R, B and D are in fiber-space, F is in ply-space
            contribution_eval
                += eval_d(diffuse) * (1.f - fresnel) * (1.f - fiber_transparency) * inter_fiber_schadowing;
            contribution_eval
                += eval_b(theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber, attenuation_b, sqrt(roughness_yarn_bm2))
                * (1.f - fresnel) * fiber_transparency * inter_fiber_schadowing;

            // Introduce falloff; we need to convert from fiber model to surface model. The wo.z is the natural thing to
            // do, but we need to compensate for the introduced darkening. Only compensate where the intershadowing is
            // applied
            contribution_eval *= abs(wo.z) / 0.5f;

            contribution_eval += eval_r(theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber, fresnel);

            contribution_eval /= max(abs(sqrt(1.f - cos_wo_tfiber * cos_wo_tfiber)), C_MIN_ANGLE);

            if (hit_top) {
                // try again for bottom fiber
                const int hit_id_bottom
                    = eval_fiber_position(wi, true, closest_hit, hit_top, intersections, hit_counter);
                const float3 weight_f = (1.f - fresnel) * fiber_transparency * attenuation_f;

                if (hit_id_bottom > -1) {
                    float3 contribution_eval_bottom = float3(0.f);
                    const float3 diffuse = eval_fiber_contribution(
                        wi,
                        closest_hit,
                        hit_id_bottom,
                        false,
                        intersections,
                        hit_counter,
                        roughness_yarn_fm2,
                        roughness_yarn_fn2,
                        attenuation_f,
                        roughness_yarn_bm2,
                        attenuation_b
                    );

                    Frame sf_yarn = get_yarn_frame(false);
                    Frame sf_ply, sf_fiber;
                    const float z = get_z(false, wi, sf_yarn.bitangent);
                    get_frames(hit_id_bottom, z, closest_hit, sf_fiber, sf_ply);

                    float3 wi_yarn, wo_yarn, wi_ply, wo_ply, wi_fiber, wo_fiber;
                    float theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber, theta_iply, theta_oply, phi_iply,
                        phi_oply, cos_wo_tfiber, cos_wo_tply;
                    get_angles(
                        wi,
                        wo,
                        sf_yarn,
                        sf_ply,
                        sf_fiber,
                        wi_yarn,
                        wo_yarn,
                        wi_ply,
                        wo_ply,
                        wi_fiber,
                        wo_fiber,
                        theta_ifiber,
                        theta_ofiber,
                        phi_ifiber,
                        phi_ofiber,
                        theta_iply,
                        theta_oply,
                        phi_iply,
                        phi_oply,
                        cos_wo_tfiber,
                        cos_wo_tply
                    );

                    const float2 uv = uv_bottom;
                    const float3 inter_fiber_schadowing = eval_local_shadowing(
                        wo,
                        closest_hit,
                        get_h(false),
                        get_z(false, wi, sf_yarn.bitangent),
                        hit_id_bottom,
                        false
                    );

                    const float fresnel = eval_fresnel_schlick(fiber_reflectance, 1.f, cos(theta_ifiber));

                    contribution_eval_bottom
                        += eval_d(diffuse) * (1.f - fresnel) * (1.f - fiber_transparency) * inter_fiber_schadowing;
                    contribution_eval_bottom += eval_b(
                                                    theta_ifiber,
                                                    theta_ofiber,
                                                    phi_ifiber,
                                                    phi_ofiber,
                                                    attenuation_b,
                                                    sqrt(roughness_yarn_bm2)
                                                )
                        * (1.f - fresnel) * fiber_transparency * inter_fiber_schadowing;

                    // intershadowing is only applied to B and D
                    contribution_eval_bottom *= abs(wo.z) / 0.5f;

                    contribution_eval_bottom += eval_r(theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber, fresnel);

                    contribution_eval_bottom /= max(abs(sqrt(1.f - cos_wo_tfiber * cos_wo_tfiber)), C_MIN_ANGLE);

                    contribution_eval_bottom += eval_f(
                                                    theta_iply,
                                                    theta_oply,
                                                    phi_iply,
                                                    phi_oply,
                                                    attenuation_f,
                                                    sqrt(roughness_yarn_fm2),
                                                    sqrt(roughness_yarn_fn2)
                                                )
                        * (1.f - fresnel) * fiber_transparency
                        / max(abs(sqrt(1.f - cos_wo_tply * cos_wo_tply)), C_MIN_ANGLE);
                    contribution_eval += contribution_eval_bottom * weight_f;
                } else {
                    contribution_eval += eval_f(
                                             theta_iply,
                                             theta_oply,
                                             phi_iply,
                                             phi_oply,
                                             attenuation_f,
                                             sqrt(roughness_yarn_fm2),
                                             sqrt(roughness_yarn_fn2)
                                         )
                        * (1.f - fresnel) * fiber_transparency
                        / max(abs(sqrt(1.f - cos_wo_tply * cos_wo_tply)), C_MIN_ANGLE);
                }
            } else
                contribution_eval += eval_f(
                                         theta_iply,
                                         theta_oply,
                                         phi_iply,
                                         phi_oply,
                                         attenuation_f,
                                         sqrt(roughness_yarn_fm2),
                                         sqrt(roughness_yarn_fn2)
                                     )
                    * (1.f - fresnel) * fiber_transparency
                    / max(abs(sqrt(1.f - cos_wo_tply * cos_wo_tply)), C_MIN_ANGLE);
        }

        if (any(isnan(contribution_eval)) || any(isinf(contribution_eval)))
            return 0.f;

        return contribution_eval;
    }

    public bool sample<S : ISampleGenerator>(
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 weight,
        out LobeTypes lobe_type,
        inout S sg,
        BSDFContext bc
    )
    {
        float2 closest_hit;
        bool hit_top;
        Intersection intersections[C_MAX_PLIES];
        int hit_counter;
        const int hit_id = eval_fiber_position(wi, false, closest_hit, hit_top, intersections, hit_counter);

        // We tested top and bottom and didn't hit anything
        if (hit_id == -1) {
            wo = float3(-wi.x, -wi.y, -wi.z);
            pdf = 0.f;
            weight = 1.f;
            lobe_type = LobeTypes::delta_transmission;
            return true;
        }

        int selected_lobe = sample_lobe(wi, closest_hit, hit_top, intersections, hit_counter, hit_id, wo, weight, sg);

        // We sampled a ray which travels into te lower hemisphere. Re-evaluate the bottom fiber if (and only if)  we
        // hit a top fiber. The incoming ray will be the outgoing ray.
        if (selected_lobe == 1 && hit_top) {
            // We know that we can only hit a bottom fiber. Set must_hit_bottom to true.
            const int hit_id = eval_fiber_position(wi, true, closest_hit, hit_top, intersections, hit_counter);

            if (hit_id != -1) {
                float3 bottom_weight;
                int selected_lobe
                    = sample_lobe(wi, closest_hit, hit_top, intersections, hit_counter, hit_id, wo, bottom_weight, sg);

                weight *= bottom_weight;
            }
        }

        pdf = eval_pdf(wi, wo, bc);
        lobe_type = (wo.z >= 0.f ? LobeTypes::glossy_reflection : LobeTypes::glossy_transmission);

        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, BSDFContext bc)
    {
        float2 closest_hit;
        bool hit_top;
        float3 roughness_yarn_fm2, roughness_yarn_fn2, roughness_yarn_bm2;
        float3 attenuation_f, attenuation_b;
        Intersection intersections[C_MAX_PLIES];
        int hit_counter;
        const int hit_id = eval_fiber_position(wi, false, closest_hit, hit_top, intersections, hit_counter);

        if (hit_id > -1) {
            const float3 diffuse = eval_fiber_contribution(
                wi,
                closest_hit,
                hit_id,
                hit_top,
                intersections,
                hit_counter,
                roughness_yarn_fm2,
                roughness_yarn_fn2,
                attenuation_f,
                roughness_yarn_bm2,
                attenuation_b
            );

            Frame sf_yarn = get_yarn_frame(hit_top);
            Frame sf_ply, sf_fiber;
            const float z = get_z(hit_top, wi, sf_yarn.bitangent);
            get_frames(hit_id, z, closest_hit, sf_fiber, sf_ply);

            // Coordinate frames and angles
            float3 wi_yarn, wo_yarn, wi_ply, wo_ply, wi_fiber, wo_fiber;
            float theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber, theta_iply, theta_oply, phi_iply, phi_oply,
                cos_wo_tfiber, cos_wo_tply;
            get_angles(
                wi,
                wo,
                sf_yarn,
                sf_ply,
                sf_fiber,
                wi_yarn,
                wo_yarn,
                wi_ply,
                wo_ply,
                wi_fiber,
                wo_fiber,
                theta_ifiber,
                theta_ofiber,
                phi_ifiber,
                phi_ofiber,
                theta_iply,
                theta_oply,
                phi_iply,
                phi_oply,
                cos_wo_tfiber,
                cos_wo_tply
            );

            const float fresnel = eval_fresnel_schlick(fiber_reflectance, 1.f, cos(theta_ifiber));
            float4 selection_probability
                = get_selection_probability(fresnel, avg(attenuation_f), avg(attenuation_b), avg(diffuse));

            float contribution_pdf = 0.f;
            contribution_pdf += pdf_r(theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber) * selection_probability[0];
            contribution_pdf += pdf_d() * selection_probability[2];
            contribution_pdf += pdf_b(theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber, sqrt(roughness_yarn_bm2))
                * selection_probability[3];

            contribution_pdf /= max(abs(sqrt(1.f - cos_wo_tfiber * cos_wo_tfiber)), C_MIN_ANGLE);

            float contribution_pdf_bottom = 0.f;
            if (hit_top) {
                // try again for bottom fiber
                const int hit_id_bottom
                    = eval_fiber_position(wi, true, closest_hit, hit_top, intersections, hit_counter);
                const float weight_f = selection_probability[1];

                if (hit_id_bottom > -1) {
                    float3 contribution_eval_bottom = float3(0.f);
                    const float3 diffuse = eval_fiber_contribution(
                        wi,
                        closest_hit,
                        hit_id_bottom,
                        false,
                        intersections,
                        hit_counter,
                        roughness_yarn_fm2,
                        roughness_yarn_fn2,
                        attenuation_f,
                        roughness_yarn_bm2,
                        attenuation_b
                    );

                    Frame sf_yarn = get_yarn_frame(false);
                    Frame sf_ply, sf_fiber;
                    const float z = get_z(false, wi, sf_yarn.bitangent);
                    get_frames(hit_id_bottom, z, closest_hit, sf_fiber, sf_ply);

                    float3 wi_yarn, wo_yarn, wi_ply, wo_ply, wi_fiber, wo_fiber;
                    float theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber, theta_iply, theta_oply, phi_iply,
                        phi_oply, cos_wo_tfiber, cos_wo_tply;
                    get_angles(
                        wi,
                        wo,
                        sf_yarn,
                        sf_ply,
                        sf_fiber,
                        wi_yarn,
                        wo_yarn,
                        wi_ply,
                        wo_ply,
                        wi_fiber,
                        wo_fiber,
                        theta_ifiber,
                        theta_ofiber,
                        phi_ifiber,
                        phi_ofiber,
                        theta_iply,
                        theta_oply,
                        phi_iply,
                        phi_oply,
                        cos_wo_tfiber,
                        cos_wo_tply
                    );

                    const float fresnel = eval_fresnel_schlick(fiber_reflectance, 1.f, cos(theta_ifiber));
                    float4 selection_probability
                        = get_selection_probability(fresnel, avg(attenuation_f), avg(attenuation_b), avg(diffuse));

                    contribution_pdf_bottom
                        += pdf_r(theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber) * selection_probability[0];
                    contribution_pdf_bottom += pdf_d() * selection_probability[2];
                    contribution_pdf_bottom
                        += pdf_b(theta_ifiber, theta_ofiber, phi_ifiber, phi_ofiber, sqrt(roughness_yarn_bm2))
                        * selection_probability[3];
                    contribution_pdf_bottom /= max(abs(sqrt(1.f - cos_wo_tfiber * cos_wo_tfiber)), C_MIN_ANGLE);

                    contribution_pdf_bottom += pdf_f(
                                                   theta_iply,
                                                   theta_oply,
                                                   phi_iply,
                                                   phi_oply,
                                                   sqrt(roughness_yarn_fm2),
                                                   sqrt(roughness_yarn_fn2)
                                               )
                        / max(abs(sqrt(1.f - cos_wo_tply * cos_wo_tply)), C_MIN_ANGLE) * selection_probability[1];

                    contribution_pdf += contribution_pdf_bottom * weight_f;
                } else {
                    contribution_pdf += pdf_f(
                                            theta_iply,
                                            theta_oply,
                                            phi_iply,
                                            phi_oply,
                                            sqrt(roughness_yarn_fm2),
                                            sqrt(roughness_yarn_fn2)
                                        )
                        / max(abs(sqrt(1.f - cos_wo_tply * cos_wo_tply)), C_MIN_ANGLE) * selection_probability[1];
                }
            } else
                contribution_pdf += pdf_f(
                                        theta_iply,
                                        theta_oply,
                                        phi_iply,
                                        phi_oply,
                                        sqrt(roughness_yarn_fm2),
                                        sqrt(roughness_yarn_fn2)
                                    )
                    / max(abs(sqrt(1.f - cos_wo_tply * cos_wo_tply)), C_MIN_ANGLE) * selection_probability[1];

            if (isnan(contribution_pdf) || isinf(contribution_pdf))
                return 0.f;

            return contribution_pdf;
        }

        // We have not hit anything and will produce a Dirac ray.
        return 0.f;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        // disregard shadowing for albedo
        float2 closest_hit;
        bool hit_top;
        float3 roughness_yarn_fm2, roughness_yarn_fn2, roughness_yarn_bm2;
        float3 attenuation_f, attenuation_b;
        Intersection intersections[C_MAX_PLIES];
        int hit_counter;
        const int hit_id = eval_fiber_position(wi, false, closest_hit, hit_top, intersections, hit_counter);

        if (hit_id > -1) {
            Frame sf_yarn = get_yarn_frame(hit_top);
            Frame sf_ply, sf_fiber;
            const float z = get_z(hit_top, wi, sf_yarn.bitangent);
            get_frames(hit_id, z, closest_hit, sf_fiber, sf_ply);

            // Coordinate frames and angles
            const float3 wi_yarn = sf_yarn.to_local(wi);
            const float3 wi_fiber = sf_fiber.to_local(wi_yarn);

            const float theta_ifiber = asin(wi_fiber.x);
            // Albedos for R should be rough Fresnel instead of perfectly specular
            const float fresnel = eval_fresnel_schlick(fiber_reflectance, 1.f, cos(theta_ifiber));
            float3 diffuse = eval_fiber_contribution(
                wi,
                closest_hit,
                hit_id,
                hit_top,
                intersections,
                hit_counter,
                roughness_yarn_fm2,
                roughness_yarn_fn2,
                attenuation_f,
                roughness_yarn_bm2,
                attenuation_b
            );
            const float3 albedo_r = fresnel + (1.f - fresnel) * diffuse + float3(closest_hit.x, closest_hit.y, z);

            // We ignore the F contribution for now and assume we don't want to layer fabric on top of something; it
            // would involve to derive an albedo for the lower hemisphere
            return AlbedoContributions(albedo_r, 1.f - albedo_r, 0.0f, 0.0f);
        } else
            return AlbedoContributions(0.0, 0.f, 1.0f, 0.0f);
    }

    public property float3 ior_as_reflectance
    {
        get {
            return fiber_reflectance;
        }
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        RoughnessInformation r;
        // r.roughness_bsdfnotation = fiber_roughness_m;

        // disregard shadowing for albedo
        float2 closest_hit;
        bool hit_top;
        float3 roughness_yarn_fm2, roughness_yarn_fn2, roughness_yarn_bm2;
        float3 attenuation_f, attenuation_b;
        Intersection intersections[C_MAX_PLIES];
        int hit_counter;
        const int hit_id = eval_fiber_position(wi, false, closest_hit, hit_top, intersections, hit_counter);

        if (hit_id > -1) {
            Frame sf_yarn = get_yarn_frame(hit_top);
            Frame sf_ply, sf_fiber;
            const float z = get_z(hit_top, wi, sf_yarn.bitangent);
            get_frames(hit_id, z, closest_hit, sf_fiber, sf_ply);

            // Coordinate frames and angles
            const float3 wi_yarn = sf_yarn.to_local(wi);
            const float3 wi_fiber = sf_fiber.to_local(wi_yarn);

            const float theta_ifiber = asin(wi_fiber.x);
            float3 diffuse = eval_fiber_contribution(
                wi,
                closest_hit,
                hit_id,
                hit_top,
                intersections,
                hit_counter,
                roughness_yarn_fm2,
                roughness_yarn_fn2,
                attenuation_f,
                roughness_yarn_bm2,
                attenuation_b
            );
            r.roughness = avg(roughness_yarn_bm2);
        } else
            r.roughness = 0.0;

        return r;
    }

    /// Private functions and math

    struct DSData
    {
        float3 ply_attenuation_m;
        float3 spread_m2;
        float3 spread_n2;
    };

    struct Intersection
    {
        int id = -1;
        float distance;
        float min_y;
    };

    void swap_intersections(inout Intersection i0, inout Intersection i1)
    {
        Intersection tmp = i0;
        i0 = i1;
        i1 = tmp;
    }

    float avg(const float3 value) { return (value[0] + value[1] + value[2]) / 3.f; }

    float erfc(const float value) { return 1.f - erf(value); }

    float phi_regularized(const float phi)
    {
        return phi < -float.PI ? phi + 2.f * float.PI : (phi > float.PI ? phi - 2.f * float.PI : phi);
    }

    /// *************************************************************************************************************
    /// Sample() and eval() of the Gaussian functions which describe the lobes
    /// ***************************************************************************************************************
    float sample_gaussian(const float2 rand, const float theta, const float roughness)
    {
        // Box-Mueller sampling; sigma = roughness; mu = theta_i
        const float roughness_safe = clamp(roughness, C_MIN_ROUGHNESS, C_MAX_ROUGHNESS);
        const float z0 = sqrt(-2.f * log(rand[0])) * cos(2.f * float.PI* rand[1]);
        return theta + z0 * roughness_safe;
    }

    float sample_truncated_gaussian(const float rand, const float limit, const float theta, const float roughness)
    {
        const float roughness_safe = clamp(roughness, C_MIN_ROUGHNESS, C_MAX_ROUGHNESS);

        const float cdf_min = cdf_gaussian(-limit, theta, roughness_safe);
        const float cdf_max = cdf_gaussian(limit, theta, roughness_safe);
        const float u = cdf_min + rand * (cdf_max - cdf_min);
        return theta - SQRT_2 * roughness_safe * erfinv(1.f - 2.f * u);
    }

    float eval_gaussian(const float theta, const float roughness)
    {
        const float roughness_safe = clamp(roughness, C_MIN_ROUGHNESS, C_MAX_ROUGHNESS);
        return 1.f / (sqrt(2.f * float.PI) * roughness_safe)
            * exp(-(theta * theta) / (2.f * roughness_safe * roughness_safe));
    }

    // Integrated Gaussian in the range [x0,x1]
    float integrate_gaussian(const float x0, const float x1, const float roughness)
    {
        const float roughness_safe = clamp(roughness, C_MIN_ROUGHNESS, C_MAX_ROUGHNESS);
        return 0.5f * (erf(x1 / (SQRT_2 * roughness_safe)) - erf(x0 / (SQRT_2 * roughness_safe)));
    }

    float cdf_gaussian(const float limit, const float theta, const float roughness)
    {
        return 0.5f * erfc((-limit + theta) / (SQRT_2 * roughness));
    }

    /// *************************************************************************************************************
    /// Dual-scattering math
    /// Dual scattering consists of two main quanitites, the attenuation and the spread (i.e. the roughness). Both are
    /// derived from the fiber properties alpha (see Yuskel et al 2008)
    /// ***************************************************************************************************************
    float3 get_alpha_forward(const float M)
    {
        const float fiber_roughness_n2 = fiber_roughness_n * fiber_roughness_n;
        float N = -2.0f * fiber_roughness_n2 * float.INV_PI * eval_gaussian(0.f, fiber_roughness_n);
        N -= -erf(float.PI / (SQRT_2 * fiber_roughness_n))
            - 2.0f * fiber_roughness_n2 * float.INV_PI * eval_gaussian(float.PI, fiber_roughness_n);

        return float3(0.5f * M, M * N, 0.5f);
    }

    float3 get_alpha_backward(const float M)
    {
        const float fiber_roughness_n2 = fiber_roughness_n * fiber_roughness_n;
        const float N = 2.0f * fiber_roughness_n2 * float.INV_PI
            * (eval_gaussian(0.f, fiber_roughness_n) - eval_gaussian(float.PI, fiber_roughness_n));
        return float3(0.5f * M, M * N, 0.5f);
    }

    void get_forward_fiber_attenuation_and_spread(
        const float M,
        const float fresnel,
        const float3 albedo,
        out float3 fiber_attenuation,
        out float3 fiber_spread_m,
        out float3 fiber_spread_n,
        out float fiber_zero_attenuation
    )
    {
        const float3 alpha_rttd = get_alpha_forward(M);
        const float3 weight_r = fresnel * alpha_rttd[0];
        float3 weight_tt = (1.f - fresnel) * fiber_transparency * alpha_rttd[1];
        float3 weight_d = (1.f - fresnel) * (1.f - fiber_transparency) * alpha_rttd[2];

        fiber_zero_attenuation = weight_r[0] + weight_tt[0] + weight_d[0];

        weight_tt *= albedo;
        weight_d *= albedo;

        const float uniform_distribution_roughness = C_UNIFORM_ROUGHNESS;
        const float3 weight_sum = weight_r + weight_tt + weight_d;
        fiber_attenuation = weight_sum;
        fiber_spread_m
            = (weight_r * fiber_roughness_m + weight_tt * fiber_roughness_m + weight_d * uniform_distribution_roughness)
            / weight_sum;
        fiber_spread_n = (weight_r * uniform_distribution_roughness + weight_tt * fiber_roughness_n
                          + weight_d * uniform_distribution_roughness)
            / weight_sum;
    }

    float3 get_forward_attenuation_m(const float3 fiber_attenuation, const float fiber_intersection_count)
    {
        return pow(fiber_attenuation, fiber_intersection_count);
    }

    float3 get_backward_attenuation_m(
        const float3 fiber_attenuation_forward,
        const float3 fiber_attenuation_backward,
        const float fiber_intersection_count
    )
    {
        const float3 fiber_attenuation_forward2 = fiber_attenuation_forward * fiber_attenuation_forward;
        const float3 fiber_attenuation_forward2_nl = pow(fiber_attenuation_forward, 2.f * fiber_intersection_count);
        const float3 one_plus_fiber_attenuation_forward23 = (-1.f + fiber_attenuation_forward2)
            * (-1.f + fiber_attenuation_forward2) * (-1.f + fiber_attenuation_forward2);
        const float3 fiber_attenuation_backward3
            = fiber_attenuation_backward * fiber_attenuation_backward * fiber_attenuation_backward;

        return (fiber_attenuation_backward * (-1.f + fiber_attenuation_forward2_nl))
            / (-1.f + fiber_attenuation_forward2)
            + (fiber_attenuation_backward3
               * (-fiber_attenuation_forward2 + pow(fiber_attenuation_forward, 4.f * fiber_intersection_count)
                  - fiber_attenuation_forward2_nl * (-1.f + fiber_attenuation_forward2)
                      * (-1.f + 2.f * fiber_intersection_count)))
            / one_plus_fiber_attenuation_forward23;
    }

    void get_backward_fiber_attenuation_and_spread(
        const float M,
        const float fresnel,
        const float3 albedo,
        out float3 fiber_attenuation,
        out float3 fiber_spread_m,
        out float3 fiber_spread_n,
        out float fiber_zero_attenuation
    )
    {
        const float3 alpha_rttd = get_alpha_backward(M);
        const float3 weight_r = fresnel * alpha_rttd[0];
        float3 weight_tt = (1.f - fresnel) * fiber_transparency * alpha_rttd[1];
        float3 weight_d = (1.f - fresnel) * (1.f - fiber_transparency) * alpha_rttd[2];

        fiber_zero_attenuation = weight_r[0] + weight_tt[0] + weight_d[0];

        weight_tt *= albedo;
        weight_d *= albedo;

        const float uniform_distribution_roughness = C_UNIFORM_ROUGHNESS;
        const float3 weight_sum = weight_r + weight_tt + weight_d;
        fiber_attenuation = weight_sum;
        fiber_spread_m
            = (weight_r * fiber_roughness_m + weight_tt * fiber_roughness_m + weight_d * uniform_distribution_roughness)
            / weight_sum;
        fiber_spread_n = (weight_r * uniform_distribution_roughness + weight_tt * fiber_roughness_n
                          + weight_d * uniform_distribution_roughness)
            / weight_sum;
    }

    void get_ply_attenuation_and_spread(
        const float theta_i,
        const float fresnel,
        const float3 albedo,
        const float l,
        out DSData forward,
        out DSData backward
    )
    {
        const float M = integrate_gaussian(-float.PI * 0.5f + theta_i, float.PI * 0.5f + theta_i, fiber_roughness_m);

        // This is in ply cooridnate space. We first calculate the average roughness of a fiber dual-scattering style.
        float3 fiber_spread_forward_m, fiber_spread_forward_n, fiber_attenuation_forward;
        float3 fiber_spread_backward_m, fiber_spread_backward_n, fiber_attenuation_backward;
        float fiber_zero_attenuation_forward, fiber_zero_attenuation_backward;
        get_forward_fiber_attenuation_and_spread(
            M,
            fresnel,
            albedo,
            fiber_attenuation_forward,
            fiber_spread_forward_m,
            fiber_spread_forward_n,
            fiber_zero_attenuation_forward
        );
        get_backward_fiber_attenuation_and_spread(
            M,
            fresnel,
            albedo,
            fiber_attenuation_backward,
            fiber_spread_backward_m,
            fiber_spread_backward_n,
            fiber_zero_attenuation_backward
        );
        const float3 fiber_attenuation_normalize = select(
            fiber_attenuation_forward + fiber_attenuation_backward > 1.f,
            fiber_attenuation_forward + fiber_attenuation_backward,
            1.f
        );
        const float3 fiber_zero_attenuation_normalize = select(
            fiber_zero_attenuation_forward + fiber_zero_attenuation_backward > 1.f,
            fiber_zero_attenuation_forward + fiber_zero_attenuation_backward,
            1.f
        );

        // compensate for twist in roughness
        // fiber_spread_forward_n = fiber_spread_forward_m * sin_theta_twist + fiber_spread_forward_n * cos_theta_twist;

        const float fiber_intersection_count = get_fiber_intersection_count(l);

        // Calculate total energy for a white fiber. We use this to make everything energy conserving because by
        // considering the first two bounces only we would loose energy on the backward component
        const float3 attenuation_normalize = get_forward_attenuation_m(
                                                 fiber_zero_attenuation_forward / fiber_zero_attenuation_normalize,
                                                 fiber_intersection_count
                                             )
            + get_backward_attenuation_m(fiber_zero_attenuation_forward / fiber_zero_attenuation_normalize,
                                         fiber_zero_attenuation_backward / fiber_zero_attenuation_normalize,
                                         fiber_intersection_count);

        // Forward lobe computations
        // For the spread we do a convolution of all the roughnesses. The roughness is the Gaussian colvolution of
        // fiber-light interactions; this should be sqrt(), but we want to have roughness^2
        forward.spread_m2 = (fiber_intersection_count * fiber_spread_forward_m * fiber_spread_forward_m);
        forward.spread_n2 = (fiber_intersection_count * fiber_spread_forward_n * fiber_spread_forward_n);
        forward.ply_attenuation_m = get_forward_attenuation_m(
                                        fiber_attenuation_forward / fiber_attenuation_normalize,
                                        fiber_intersection_count
                                    )
            / attenuation_normalize;

        // Backward lobe computations
        backward.ply_attenuation_m = get_backward_attenuation_m(
                                         fiber_attenuation_forward / fiber_attenuation_normalize,
                                         fiber_attenuation_backward / fiber_attenuation_normalize,
                                         fiber_intersection_count
                                     )
            / attenuation_normalize;

        const float3 beta_bm2 = fiber_spread_backward_m * fiber_spread_backward_m;
        const float3 beta_fm2 = fiber_spread_forward_m * fiber_spread_forward_m;
        const float3 fiber_attenuation_bnormalized = fiber_attenuation_backward / fiber_attenuation_normalize;
        const float3 fiber_attenuation_backward3
            = fiber_attenuation_bnormalized * fiber_attenuation_bnormalized * fiber_attenuation_bnormalized;
        // This is the correct solution, but can become quite long for forward scattering materials. We will approximate
        // it with the infinite approximative formula without the density compensation.
        ///         const float3 three_beta_bm2 = 3.f * fiber_spread_backward_m*fiber_spread_backward_m;
        /// float spread3_m = 0.f;
        /// for (int n = 1; n <= fiber_count; n++)
        /// {
        ///     const float x = spread3_m;
        ///     spread3_m += (n*(n+1.f) * 0.5f * af2n * sqrt(three_beta_bm2 + (n*2.f)*beta_fm2));
        ///     if (spread3_m - x < 0.001f) break;
        /// }
        /// const float spread1_m = (((-1.f + pow(fiber_attenuation_forward, 2.f *fiber_count)) * sqrt(beta_bm2
        /// + 2.f*beta_fm2])/(-1.f + fiber_attenuation_bnormalized * fiber_attenuation_bnormalized)) ;
        /// backward.spread_m2 = pow( fiber_attenuation_bnormalized / backward.ply_attenuation_m * spread1_m +
        /// fiber_attenuation_bnormalized / backward.ply_attenuation_m * spread3_m , 2);
        backward.spread_m2 = pow(
            (fiber_attenuation_bnormalized * sqrt(beta_bm2 + 2.f * beta_fm2)
             + fiber_attenuation_backward3 * sqrt(3.f * beta_bm2 + 2.f * beta_fm2))
                / (fiber_attenuation_backward
                   + fiber_attenuation_backward3 * (3.f * fiber_spread_backward_m + 2.f * fiber_spread_forward_m)),
            2
        );
        backward.spread_n2 = 0.0f; // unused since not Gaussian
    }

    /// *************************************************************************************************************
    ///  R lobe. Fiber coordinate space.
    /// ****************************************************************************************************************
    float2 sample_r(const float3 rand, const float theta_i, inout float3 weight)
    {
        const float theta_o = sample_truncated_gaussian(rand.x, float.PI * 0.5f, -theta_i, fiber_roughness_m);
        const float phi_o = (rand[2] * (2.f * float.PI) - float.PI) * 0.5f; // random number in [-pi/2, pi/2] range]

        weight = eval_fresnel_schlick(fiber_reflectance, 1.f, cos(theta_i));
        return float2(theta_o, phi_o);
    }

    float3 eval_r(const float theta_i, const float theta_o, const float phi_i, const float phi_o, const float fresnel)
    {
        const float m_r = abs(theta_o) < float.PI * 0.5f ? eval_gaussian(theta_o + theta_i, fiber_roughness_m) : 0.f;
        const float n_r = abs(phi_regularized(phi_o + phi_i + float.PI)) < float.PI * 0.5f ? 1.f / float.PI : 0.f;
        const float normalize_gaussian
            = integrate_gaussian(-float.PI * 0.5f + (-theta_i), float.PI * 0.5f + (-theta_i), fiber_roughness_m);
        return fresnel * m_r * n_r / normalize_gaussian;
    }

    float pdf_r(const float theta_i, const float theta_o, const float phi_i, const float phi_o, )
    {
        const float m_r = abs(theta_o) < float.PI * 0.5f ? eval_gaussian(theta_o + theta_i, fiber_roughness_m) : 0.f;
        const float n_r = abs(phi_regularized(phi_o + phi_i + float.PI)) < float.PI * 0.5f ? 1.f / float.PI : 0.f;
        const float normalize_gaussian
            = integrate_gaussian(-float.PI * 0.5f + (-theta_i), float.PI * 0.5f + (-theta_i), fiber_roughness_m);
        return m_r * n_r / normalize_gaussian;
    }

    /// *************************************************************************************************************
    ///  F lobe
    /// ****************************************************************************************************************
    float2 sample_f(
        const float2 rand,
        const float2 rand2,
        const float theta_i,
        const float phi_i,
        const float3 aggregate_roughness_m,
        const float3 aggregate_roughness_n,
        inout float3 weight
    )
    {

        const float theta_o = sample_gaussian(rand.xy, -theta_i, avg(aggregate_roughness_m));
        const float phi_o = sample_gaussian(rand2.xy, phi_i + float.PI, avg(aggregate_roughness_n));

        weight = 1.0f;
        return float2(theta_o, phi_o);
    }

    float3 eval_f(
        const float theta_i,
        const float theta_o,
        const float phi_i,
        const float phi_o,
        const float3 attenuation_f,
        const float3 roughness_yarn_fm,
        const float3 roughness_yarn_fn
    )
    {
        const float3 m_f = eval_gaussian(theta_o + theta_i, avg(roughness_yarn_fm));
        float phi_d = phi_regularized(phi_o - (phi_i + float.PI));
        const float3 n_f = eval_gaussian(phi_d, avg(roughness_yarn_fn));
        return m_f * n_f * attenuation_f;
    }

    float pdf_f(
        const float theta_i,
        const float theta_o,
        const float phi_i,
        const float phi_o,
        const float3 roughness_yarn_fm,
        const float3 roughness_yarn_fn
    )
    {
        const float m_f = eval_gaussian(theta_o + theta_i, avg(roughness_yarn_fm));
        float phi_d = phi_regularized(phi_o - (phi_i + float.PI));
        const float n_f = eval_gaussian(phi_d, avg(roughness_yarn_fn));
        return m_f * n_f;
    }

    /// *************************************************************************************************************
    ///  B lobe
    /// ****************************************************************************************************************
    float2 sample_b(
        const float3 rand,
        const float theta_i,
        const float phi_i,
        const float3 roughness_yarn_bm,
        inout float3 weight
    )
    {

        const float theta_o = sample_truncated_gaussian(rand.x, float.PI * 0.5f, -theta_i, avg(roughness_yarn_bm));
        const float phi_o = (rand[2] * (2.f * float.PI) - float.PI) * 0.5; // random number in [-pi/2, pi,2] range]

        weight = 1.0f;
        return float2(theta_o, phi_o);
    }

    float3 eval_b(
        const float theta_i,
        const float theta_o,
        const float phi_i,
        const float phi_o,
        const float3 attenuation_b,
        const float3 roughness_yarn_bm
    )
    {

        const float m_b
            = abs(theta_o) < float.PI * 0.5f ? eval_gaussian(theta_o + theta_i, avg(roughness_yarn_bm)) : 0.f;
        const float n_b = abs(phi_regularized(phi_o + phi_i + float.PI)) < float.PI * 0.5 ? 1.f / float.PI : 0.f;
        const float normalize_gaussian
            = integrate_gaussian(-float.PI * 0.5f + (-theta_i), float.PI * 0.5f + (-theta_i), avg(roughness_yarn_bm));
        return m_b * n_b * attenuation_b / normalize_gaussian;
    }

    float pdf_b(
        const float theta_i,
        const float theta_o,
        const float phi_i,
        const float phi_o,
        const float3 roughness_yarn_bm
    )
    {
        const float m_b = eval_gaussian(theta_o + theta_i, avg(roughness_yarn_bm));
        const float n_b = abs(phi_regularized(phi_o + phi_i + float.PI)) < float.PI * 0.5 ? 1.f / float.PI : 0.f;
        return m_b * n_b;
    }

    /// *************************************************************************************************************
    ///  D lobe
    /// ****************************************************************************************************************
    float2 sample_d(const float2 rand, const float theta_i, inout float3 weight)
    {
        // this is in fiber cooridnate space
        const float theta_o = rand[0] * float.PI - (float.PI * 0.5);
        const float phi_o = (rand[1] * (2.f * float.PI) - float.PI); // random number in [-pi, pi] range]

        weight = 1.0f;
        return float2(theta_o, phi_o);
    }

    float3 eval_d(const float3 diffuse)
    {
        const float m_d = 1.f / float.PI;
        const float n_d = 0.5f / float.PI;
        return m_d * n_d * diffuse;
    }

    float pdf_d()
    {
        const float m_d = 1.f / float.PI;
        const float n_d = 0.5f / float.PI;
        return m_d * n_d;
    }

    /*****************************************************************************************************************
     *  Helper functions
     ******************************************************************************************************************/
    // Position along the yarn
    float get_z(const bool hit_top, const float3 wi, const float3 yarn_biangent)
    {
        return (hit_top ? uv_top.x : uv_bottom.x) - acos(dot(wi, yarn_biangent)) * (ply_twist < 0.f ? -1.f : 1.f);
    }

    // Position accross the yarn
    float get_h(const bool hit_top)
    {
        const float position = hit_top ? uv_top.y : uv_bottom.y;

        float temp_workaround;
        if (hit_top && yarn_id == 0)
            temp_workaround = yarn_scale.y;
        else if (hit_top && yarn_id == 1)
            temp_workaround = yarn_scale.x;
        else if (!hit_top && yarn_id == 0)
            temp_workaround = yarn_scale.x;
        else
            temp_workaround = yarn_scale.y;

        return (position * 2.0f - 1.0f) * temp_workaround;
    }

    float get_ply_rotation(const int id_ply, const float z)
    {
        return 2.f * float.PI * (1.f / ply_count * id_ply + z * ply_twist);
    }

    float get_ply_diameter() { return sqrt(1.f / ply_count) * 2.f * ply_scale; }

    float get_fiber_radius()
    {
        // Average size of fiber for a completely densly packed circle. We can add sparness here if we want.
        return sqrt(1.f / fiber_count);
    }

    float2 rotate2_d(const float2 p, const float a)
    {
        const float sin_rotation = sin(a);
        const float cos_rotation = cos(a);
        return float2(p.x * cos_rotation - p.y * sin_rotation, p.x * sin_rotation + p.y * cos_rotation);
    }

    // The cross section of a helix is not a circle but an ellipse.
    bool ellipse_line_intersection(
        const float2 p,
        const float2 d,
        const float2 ply_position_center,
        const float2 axis,
        const float rotation_angle,
        inout float2 instersect_position,
        inout float2 instersect_position2
    )
    {
        const float2 rotated_line_point = rotate2_d(p - ply_position_center, -rotation_angle);
        const float2 rotated_line_direction = rotate2_d(d, -rotation_angle);
        const float axis_x2 = axis.x * axis.x;
        const float axis_y2 = axis.y * axis.y;

        const float a = rotated_line_direction.x * rotated_line_direction.x / axis_x2
            + rotated_line_direction.y * rotated_line_direction.y / axis_y2;
        const float b = 2.f
            * (rotated_line_point.x * rotated_line_direction.x / axis_x2
               + rotated_line_point.y * rotated_line_direction.y / axis_y2);
        const float c = (rotated_line_point.x * rotated_line_point.x / axis_x2
                         + rotated_line_point.y * rotated_line_point.y / axis_y2)
            - 1.f;

        const float disc = b * b - 4.f * a * c;

        if (disc >= 0.f) {
            // resulting two intersections for front and back; we need the first one to know which ply we hit and the
            // second one for the distance in the ply
            const float t1 = (-b + sqrt(disc)) / (2.f * a);
            const float t2 = (-b - sqrt(disc)) / (2.f * a);
            instersect_position
                = rotate2_d(rotated_line_point + t1 * rotated_line_direction, rotation_angle) + ply_position_center;
            instersect_position2
                = rotate2_d(rotated_line_point + t2 * rotated_line_direction, rotation_angle) + ply_position_center;

            return true;
        } else
            return false;
    }

    // Intersect a ply with a line. The cross section of a helix is not a circle but an ellipse. We use the helix angle
    // to prohject the circle into an ellipse.
    bool intersect_ply(
        const int id_ply,
        const float h,
        const float z,
        const float2 start,
        const float2 end,
        const bool is_circle,
        inout float2 instersect_position,
        inout float2 instersect_position2
    )
    {
        const float ply_diameter = get_ply_diameter();

        // add rotation of ply
        const float ply_rotation = get_ply_rotation(id_ply, z);

        // calculate the position of each ply
        const float2 startposition = float2(-1.0f + ply_diameter * 0.5, 0.f);
        const float2 ply_position_center = rotate2_d(startposition, ply_rotation);
        const float2 ply_position_center180 = rotate2_d(startposition, float.PI);

        // Project circle into helix plane; we take the half pitch of the helix to find the angle.
        const float3 twist_dir = ply_twist != 0.f
            ? normalize(float3(ply_position_center180, 1.f / (ply_twist * 0.5f)) - float3(ply_position_center, 0.f))
            : 1.f;
        const float axis_x = ply_diameter * 0.5f * (is_circle ? 1.f : twist_dir.z);
        const float axis_y = ply_diameter * 0.5f;

        return ellipse_line_intersection(
            start,
            normalize(end - start),
            ply_position_center,
            float2(axis_x, axis_y),
            ply_rotation,
            instersect_position,
            instersect_position2
        );
    }

    // Number of fibers a ray of length l will intersect on average while passing through a circle of radius 1; twist
    // compensation is removed for now
    float get_fiber_intersection_count(const float l)
    {
        // Simplified from the paper
        // const float r = get_fiber_radius();
        // return sqrt(fiber_count / float.PI) * l * fiber_count * float.PI * r * r / float.PI;
        return sqrt(fiber_count) * l / 1.77245f;
    }

    /// *************************************************************************************************************
    /// Coordinate frame calculations; we have three different coordinate frames, yarn, ply and fiber
    /// Calculate coordinate frame for ply and fiber. Each ply has a local coordinate frame along (1,0,0)
    /// where ply computations are done
    /// ****************************************************************************************************************
    Frame get_yarn_frame(const bool hit_top)
    {
        const float4 tangent_yarn = hit_top
            ? (yarn_direction_top.x > yarn_direction_top.y ? -1.f : 1.f) * float4(yarn_direction_top, -1.f)
            : (yarn_direction_bottom.x > yarn_direction_bottom.y ? -1.f : 1.f) * float4(yarn_direction_bottom, -1.f);

        const float3 bitangent_yarn = normalize(cross(float3(0.f, 0.f, 1.f), tangent_yarn.xyz));
        const float3 normal_yarn = normalize(cross(tangent_yarn.xyz, bitangent_yarn));

        var result = Frame(normal_yarn, tangent_yarn.xyz);
        result.bitangent *= tangent_yarn.w;
        return result;
    }

    // Calculate coordinate frame for ply and fiber. Each ply has a local coordinate frame along (1,0,0) where ply
    // computations are done
    void get_frames(const int id_ply, const float z, const float2 hit_pos, inout Frame sf_fiber, inout Frame sf_ply)
    {
        const float ply_diameter = get_ply_diameter();

        // calculate direction of a ply; this is the tangent of a helix
        const float ply_rotation = get_ply_rotation(id_ply, z);
        const float2 startposition = float2(-1.0f + ply_diameter * 0.5, 0.f);
        const float2 ply_position_center = rotate2_d(startposition, ply_rotation);
        const float2 ply_position_center180 = rotate2_d(startposition, float.PI);

        // Project circle into helix plane; we take the half pitch of the helix to find the angle.
        const float3 twist_dir = ply_twist != 0.f
            ? normalize(float3(ply_position_center180, 1.f / (ply_twist * 0.5f)) - float3(startposition, 0.f))
            : 1.0f;

        const float t = ply_rotation % (2.f * float.PI);

        // Ply coordinate frame
        // const float r = -startposition.x / yarn_scale;
        const float r = -startposition.x;
        const float3 tangent_ply
            = (ply_twist != 0.f ? normalize(float3(1.f / ply_twist, -r * sin(t), r * cos(t))) : float3(1.f, 0.f, 0.f));
        float3 normal_ply = float3(0.f, normalize(hit_pos - ply_position_center));
        const float3 bitangent_ply = normalize(cross(normal_ply, tangent_ply));
        normal_ply = cross(tangent_ply, bitangent_ply);

        // Fiber coordinate frame
        const float3 fiber_twist = fiber_twist_angle + (1.f - abs(twist_dir.z)) * sign(twist_dir.z);
        float3 normal_fiber = normal_ply;
        const float3 tangent_fiber = normalize(float3(
            tangent_ply * cos(fiber_twist) + bitangent_ply * sin(fiber_twist)
            + normal_ply * dot(normal_ply, tangent_ply) * (1.f - cos(fiber_twist))
        ));
        const float3 bitangent_fiber = normalize(cross(normal_fiber, tangent_fiber));
        normal_fiber = normalize(cross(tangent_fiber, bitangent_fiber));

        sf_fiber = Frame(normal_fiber, tangent_fiber);
        sf_fiber.bitangent *= -1;
        sf_ply = Frame(normal_ply, tangent_ply);

        return;
    }

    void get_angles(
        const float3 wi,
        const float3 wo,
        const Frame sf_yarn,
        const Frame sf_ply,
        const Frame sf_fiber,
        out float3 wi_yarn,
        out float3 wo_yarn,
        out float3 wi_ply,
        out float3 wo_ply,
        out float3 wi_fiber,
        out float3 wo_fiber,
        out float theta_ifiber,
        out float theta_ofiber,
        out float phi_ifiber,
        out float phi_ofiber,
        out float theta_iply,
        out float theta_oply,
        out float phi_iply,
        out float phi_oply,
        out float cos_wo_tfiber,
        out float cos_wo_tply
    )
    {
        wi_yarn = sf_yarn.to_local(wi);
        wo_yarn = sf_yarn.to_local(wo);
        wi_ply = sf_ply.to_local(wi_yarn);
        wo_ply = sf_ply.to_local(wo_yarn);
        wi_fiber = sf_fiber.to_local(wi_yarn);
        wo_fiber = sf_fiber.to_local(wo_yarn);

        theta_ifiber = asin(wi_fiber.x);
        theta_ofiber = asin(wo_fiber.x);
        phi_ifiber = atan2(wi_fiber.z, wi_fiber.y);
        phi_ofiber = atan2(wo_fiber.z, wo_fiber.y);
        theta_iply = asin(wi_ply.x);
        theta_oply = asin(wo_ply.x);
        phi_iply = atan2(wi_ply.z, wi_ply.y);
        phi_oply = atan2(wo_ply.z, wo_ply.y);

        cos_wo_tfiber = clamp(dot(sf_yarn.to_global(sf_fiber.tangent), wo), -1.f, 1.f);
        cos_wo_tply = clamp(dot(sf_yarn.to_global(sf_ply.tangent), wo), -1.f, 1.f);
    }

    /*****************************************************************************************************************
     *  Shading computations
     ******************************************************************************************************************/
    // Calculate the shadowing from one ply to the other. This is applied to the B and D lobes.
    float3 eval_local_shadowing(
        const float3 dir,
        const float2 closest_hit,
        const float h,
        const float z,
        const int id_ply_hit,
        const bool hit_top
    )
    {
        const float ply_diameter = get_ply_diameter();

        // todo: replace with real fiber direction
        Frame sf_yarn = get_yarn_frame(hit_top);
        float3 dir_yarn_space = sf_yarn.to_local(dir);
        dir_yarn_space = normalize(float3(0.0, dir_yarn_space.y, dir_yarn_space.z));

        const float2 start = closest_hit;
        const float2 end = closest_hit + dir_yarn_space.yz * float2(3.f);

        const float ply_rotation = get_ply_rotation(id_ply_hit, z);

        const float2 startposition = float2(-1.0f + ply_diameter * 0.5, 0.f);
        const float2 ply_position_center = rotate2_d(startposition, ply_rotation);

        Frame sf_ply, sf_fiber;
        get_frames(id_ply_hit, z, closest_hit, sf_fiber, sf_ply);

        const float3 normal = sf_yarn.to_global(sf_ply.normal);

        for (int i = 0; i < ply_count; i++) {
            if (id_ply_hit == i)
                continue;

            float2 hit_pos, hit_pos2;
            const bool hit = intersect_ply(i, h, z, start, end, true, hit_pos, hit_pos2);

            if (hit) {
                if ((dot(normal, dir) > 0.f) && (dot(normalize(hit_pos - start), dir_yarn_space.yz) > 0.f)) {
                    return float3(0.0);
                }
            }
        }

        return 1.0;
    }

    // Evaluate which fiber (if any) is hit. If no fiber is hit, we return -1, otherwise we return the ID of the fiber
    // we hit. We also return a list of (sorted) intersections.
    int eval_fiber_position(
        const float3 dir,
        const bool must_hit_bottom,
        out float2 closest_hit,
        out bool hit_top,
        out Intersection intersections[C_MAX_PLIES],
        out int hit_counter
    )
    {
        // azimuthal offset on fiber; cooridnates are 0-1 in one direction and 0-n in the other
        const float h = get_h(true);

        // trace against imaginary curve
        // iterate over all plies and check which one we hit. We only take the closest intersection, but we need all
        // intersections for the F lobe initialize outside the circle
        closest_hit = { 0.0, -3.f };
        hit_top = false;
        hit_counter = 0;
        int hit_id = -1;

        if (!must_hit_bottom) {
            Frame sf_yarn = get_yarn_frame(true);
            for (int i = 0; i < ply_count; i++) {
                float2 hit_pos, hit_pos2;
                const bool hit = intersect_ply(
                    i,
                    h,
                    get_z(true, dir, sf_yarn.bitangent),
                    float2(h, -2.f),
                    float2(h, 2.f),
                    false,
                    hit_pos,
                    hit_pos2
                );

                if (hit) {
                    if (hit_pos.y > closest_hit.y) {
                        closest_hit = hit_pos;
                        hit_id = i;
                    }

                    intersections[hit_counter].id = i;
                    intersections[hit_counter].distance = length(hit_pos2 - hit_pos);
                    intersections[hit_counter].min_y = hit_pos.y;

                    hit_top = true;
                    hit_counter++;
                }
            }
        }

        if (hit_id == -1) {
            // we hit a transparent area; this can either mean that there is nothing or that we have hit the underlying
            // fiber through the top fiber test again, but this time the bottom fiber
            const float h = get_h(false);

            Frame sf_yarn = get_yarn_frame(false);
            for (int i = 0; i < ply_count; i++) {
                float2 hit_pos, hit_pos2;
                const bool hit = intersect_ply(
                    i,
                    h,
                    get_z(false, dir, sf_yarn.bitangent),
                    float2(h, -2.f),
                    float2(h, 2.f),
                    false,
                    hit_pos,
                    hit_pos2
                );

                if (hit) {
                    intersections[hit_counter].id = i;
                    intersections[hit_counter].distance = length(hit_pos2 - hit_pos);
                    intersections[hit_counter].min_y = hit_pos.y;

                    if (hit_pos.y > closest_hit.y) {
                        closest_hit = hit_pos;
                        hit_id = i;
                    }

                    hit_top = false;
                    hit_counter++;
                }
            }
        }

        bool swapped = false;
        // Sort intersections according to their minimum y
        for (int i = 0; i < hit_counter - 1; i++) {
            swapped = false;
            for (int j = 0; j < hit_counter - i - 1; j++) {
                if (intersections[j].min_y < intersections[j + 1].min_y) {
                    swap_intersections(intersections[j], intersections[j + 1]);
                    swapped = true;
                }
            }
            if (swapped == false)
                break;
        }

        return hit_id;
    }

    // Calculate the dual scattering contributions for B and F and the diffuse color for D
    float3 eval_fiber_contribution(
        const float3 wi,
        const float2 closest_hit,
        const int hit_id,
        const bool hit_top,
        const Intersection intersections[C_MAX_PLIES],
        const int hit_counter,
        out float3 roughness_yarn_fm2,
        out float3 roughness_yarn_fn2,
        out float3 attenuation_f,
        out float3 roughness_yarn_bm2,
        out float3 attenuation_b
    )
    {
        if (hit_id == -1) {
            roughness_yarn_fm2 = 0.f;
            roughness_yarn_fn2 = 0.f;
            attenuation_f = 0.f;
            roughness_yarn_bm2 = 0.f;
            attenuation_b = 0.f;
            return 0.0;
        }

        float3 t_f = 1.f;
        float3 t_b = 0.f;
        roughness_yarn_fm2 = 0.0f;
        roughness_yarn_fn2 = 0.0f;
        roughness_yarn_bm2 = 0.0f;

        Frame sf_yarn = get_yarn_frame(hit_top);
        const float3 wi_yarn = sf_yarn.to_local(wi);

        const float3 fiber_color = (hit_top ? fiber_color_top : fiber_color_bottom);
        const float z = get_z(hit_top, wi, sf_yarn.bitangent);
        for (int i = 0; i < hit_counter; i++) {
            Frame sf_ply, sf_fiber;
            get_frames(intersections[i].id, z, closest_hit, sf_fiber, sf_ply);

            DSData forward, backward;
            // for every hit, get distance within ply and compute spread and attenuation
            const float distance = intersections[i].distance;

            // calculate attenuation and spread for F and B lobes
            float3 wi_ply = sf_ply.to_local(wi_yarn);
            const float theta_i = asin(wi_ply.x);
            const float fresnel = eval_fresnel_schlick(fiber_reflectance, 1.f, cos(theta_i));
            get_ply_attenuation_and_spread(theta_i, fresnel, fiber_color, distance, forward, backward);

            // B lobe;
            roughness_yarn_bm2 += t_f * t_f * forward.spread_m2;
            t_b += (backward.ply_attenuation_m) * t_f * t_f;

            // F lobe;
            roughness_yarn_fm2 += forward.ply_attenuation_m * forward.spread_m2;
            roughness_yarn_fn2 += forward.ply_attenuation_m * forward.spread_n2;
            t_f *= forward.ply_attenuation_m;
        }

        attenuation_f = t_f;
        attenuation_b = t_b;

        return fiber_color;
    }

    float4 get_selection_probability(
        const float intensity_r,
        const float intensity_f,
        const float intensity_b,
        const float intensity_d
    )
    {
        float4 selection_weights = float4(
            intensity_r,
            intensity_f * (1.f - intensity_r) * fiber_transparency,
            intensity_b * (1.f - intensity_r) * fiber_transparency,
            intensity_d * (1.f - intensity_r) * (1.f - fiber_transparency)
        );
        // float4 selection_weights = float4(1,0,0,0);
        const float selection_weights_norm
            = selection_weights[0] + selection_weights[1] + selection_weights[2] + selection_weights[3];
        selection_weights = selection_weights_norm > 0.f ? selection_weights / selection_weights_norm : 0.f;

        return selection_weights;
    }

    // Sample one of the lobe according to it's intensity.
    int sample_lobe<S : ISampleGenerator>(
        const float3 wi,
        const float2 closest_hit,
        const bool hit_top,
        const Intersection intersections[C_MAX_PLIES],
        const int hit_counter,
        const int hit_id,
        out float3 wo,
        out float3 weight,
        inout S sg
    )
    {
        Frame sf_yarn = get_yarn_frame(hit_top);
        Frame sf_ply, sf_fiber;
        const float z = get_z(hit_top, wi, sf_yarn.bitangent);
        get_frames(hit_id, z, closest_hit, sf_fiber, sf_ply);

        const float3 wi_yarn = sf_yarn.to_local(wi);
        float3 wi_ply = sf_ply.to_local(wi_yarn);
        float3 wi_fiber = sf_fiber.to_local(wi_yarn);
        const float theta_ifiber = asin(wi_fiber.x);

        float3 roughness_yarn_fm2, roughness_yarn_fn2, roughness_yarn_bm2;
        float3 attenuation_f, attenuation_b;

        const float fresnel = eval_fresnel_schlick(fiber_reflectance, 1.f, cos(theta_ifiber));
        const float3 diffuse = eval_fiber_contribution(
            wi,
            closest_hit,
            hit_id,
            hit_top,
            intersections,
            hit_counter,
            roughness_yarn_fm2,
            roughness_yarn_fn2,
            attenuation_f,
            roughness_yarn_bm2,
            attenuation_b
        );

        // ideally we would use something like hero-sampling to reduce noise
        float4 selection_weights
            = get_selection_probability(fresnel, avg(attenuation_f), avg(attenuation_b), avg(diffuse));

        const float rand_select_lobe = next_float(sg);
        int selected_lobe;
        if (rand_select_lobe < selection_weights[0]) // R lobe
        {
            float phi_i = atan2(wi_fiber.z, wi_fiber.y);
            float2 theta_phi = sample_r(next_float3(sg), theta_ifiber, weight);

            float sin_theta_o = sin(theta_phi[0]);
            float cos_theta_o = cos(theta_phi[0]);
            float phi_o = phi_i + float.PI + theta_phi[1];

            float3 wo_fiber = float3(sin_theta_o, cos_theta_o * cos(phi_o), -cos_theta_o * sin(phi_o)); // roble -> why
                                                                                                        // is sign
                                                                                                        // different?
            wo = sf_yarn.to_global(sf_fiber.to_global(wo_fiber));

            weight = weight / selection_weights[0];
            selected_lobe = 0;
        } else if (rand_select_lobe >= selection_weights[0]
                   && rand_select_lobe < selection_weights[0] + selection_weights[1]) {
            // F lobe
            float phi_i = atan2(wi_ply.z, wi_ply.y);
            float2 theta_phi = sample_f(
                next_float2(sg),
                next_float2(sg),
                asin(wi_ply.x),
                phi_i,
                sqrt(roughness_yarn_fm2),
                sqrt(roughness_yarn_fn2),
                weight
            );
            weight = attenuation_f;

            float sin_theta_o = sin(theta_phi[0]);
            float cos_theta_o = cos(theta_phi[0]);
            float phi_o = theta_phi[1];
            float3 wo_fiber = float3(sin_theta_o, cos_theta_o * cos(phi_o), cos_theta_o * sin(phi_o));
            wo = sf_yarn.to_global(sf_ply.to_global(wo_fiber));

            weight = weight / selection_weights[1] * (1.f - fresnel) * fiber_transparency;
            selected_lobe = 1;
        } else if (rand_select_lobe >= selection_weights[0] + selection_weights[1]
                   && rand_select_lobe < selection_weights[0] + selection_weights[1] + selection_weights[2]) {
            // B lobe
            float phi_i = atan2(wi_fiber.z, wi_fiber.y);
            float2 theta_phi = sample_b(next_float3(sg), asin(wi_fiber.x), phi_i, sqrt(roughness_yarn_bm2), weight);
            weight = attenuation_b;

            float sin_theta_o = sin(theta_phi[0]);
            float cos_theta_o = cos(theta_phi[0]);
            float phi_o = phi_i + float.PI + theta_phi[1];

            float3 wo_fiber = float3(sin_theta_o, cos_theta_o * cos(phi_o), -cos_theta_o * sin(phi_o));

            wo = sf_yarn.to_global(sf_fiber.to_global(wo_fiber));

            weight = weight / (selection_weights[2]) * (1.f - fresnel) * fiber_transparency;
            selected_lobe = 2;
        } else {
            // D lobe
            float phi_i = atan2(wi_fiber.z, wi_fiber.y);
            float2 theta_phi = sample_d(next_float2(sg), asin(wi_fiber.x), weight);

            float sin_theta_o = sin(theta_phi[0]);
            float cos_theta_o = cos(theta_phi[0]);
            float phi_o = theta_phi[1];

            float3 wo_fiber = float3(sin_theta_o, cos_theta_o * cos(phi_o), cos_theta_o * sin(phi_o));

            wo = sf_yarn.to_global(sf_fiber.to_global(wo_fiber));

            weight = diffuse / (selection_weights[3]) * (1.f - fresnel) * (1.f - fiber_transparency);
            selected_lobe = 3;
        }

        if (selected_lobe == 2 || selected_lobe == 3) {
            // azimuthal offset on fiber; cooridnates are 0-1 in one direction and 0-n in the other
            // we only apply the shadoing on the B and D component
            const float3 inter_fiber_schadowing = eval_local_shadowing(
                wo,
                closest_hit,
                get_h(hit_top),
                get_z(hit_top, wi, sf_yarn.bitangent),
                hit_id,
                hit_top
            );

            weight *= inter_fiber_schadowing;
            weight *= abs(wo.z) / 0.5f;
        }

        return selected_lobe;
    }
};
