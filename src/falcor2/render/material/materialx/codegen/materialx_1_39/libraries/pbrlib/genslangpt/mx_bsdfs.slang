// SPDX-License-Identifier: Apache-2.0

#ifdef MX139_STACK_HAS_BSDF_FRAMES
void mx_store_bsdf_frame(BSDF bsdf)
{
    const int bsdf_index = bsdf.get_index();
    mx_stack.bsdf_frames[bsdf_index] = Frame(mx_stack.bsdf_n[bsdf_index], mx_stack.bsdf_t[bsdf_index]);
}

void mx_store_curve_bsdf_frame(BSDF bsdf)
{
    const int bsdf_index = bsdf.get_index();
    mx_stack.bsdf_frames[bsdf_index]
        = Frame(mx_stack.bsdf_n[bsdf_index], mx_stack.bsdf_t[bsdf_index], Frame::Component::tangent);
}
#endif

void mx_prepare_bsdf(BSDF bsdf, float3 normal, float3 tangent)
{
    mx_stack.bsdf_n[bsdf.get_index()] = adjust_normal(normal);
    mx_stack.bsdf_t[bsdf.get_index()] = si.shading_frame_ws.to_local(tangent);
    orthonormalize(mx_stack.bsdf_n[bsdf.get_index()], mx_stack.bsdf_t[bsdf.get_index()]);
#ifdef MX139_STACK_HAS_BSDF_FRAMES
    mx_store_bsdf_frame(bsdf);
#endif
}

void mx_prepare_bsdf_with_default_tangent(BSDF bsdf, float3 normal)
{
    mx_stack.bsdf_n[bsdf.get_index()] = adjust_normal(normal);
    // Match the 1.38 MaterialX path: default explicit-normal lobes inherit the
    // surface tangent, then orthonormalize it against the BSDF normal.
    mx_stack.bsdf_t[bsdf.get_index()] = si.shading_frame_ws.to_local(si.shading_frame_ws.tangent);
    orthonormalize(mx_stack.bsdf_n[bsdf.get_index()], mx_stack.bsdf_t[bsdf.get_index()]);
#ifdef MX139_STACK_HAS_BSDF_FRAMES
    mx_store_bsdf_frame(bsdf);
#endif
}

float3 mx_default_tangent_from_normal(float3 normal)
{
    const float sign = select(normal.z < 0.0, -1.0, 1.0);
    const float a = -1.0 / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    return float3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
}

void mx_prepare_subsurface_diffuse_fallback_bsdf(BSDF bsdf, float3 normal)
{
    // MX_SUBSURFACE_ID uses MX139's diffuse fallback. Prepare a stable
    // normal-derived tangent while keeping the BSDF itself local.
    const float3 n_ws = normalize(normal);
    mx_stack.bsdf_n[bsdf.get_index()] = normalize(si.shading_frame_ws.to_local(n_ws));
    mx_stack.bsdf_t[bsdf.get_index()] = si.shading_frame_ws.to_local(mx_default_tangent_from_normal(n_ws));
    orthonormalize(mx_stack.bsdf_n[bsdf.get_index()], mx_stack.bsdf_t[bsdf.get_index()]);
#ifdef MX139_STACK_HAS_BSDF_FRAMES
    mx_store_bsdf_frame(bsdf);
#endif
}

void mx_prepare_curve_bsdf(BSDF bsdf, float3 normal, float3 curve_direction)
{
    mx_stack.bsdf_n[bsdf.get_index()] = adjust_normal(normal);
    mx_stack.bsdf_t[bsdf.get_index()] = normalize(si.shading_frame_ws.to_local(curve_direction));
#ifdef MX139_STACK_HAS_BSDF_FRAMES
    mx_store_curve_bsdf_frame(bsdf);
#endif
}

void mx_oren_nayar_diffuse_bsdf(
    ClosureData closure_data,
    float weight,
    float3 color,
    float roughness,
    float3 normal,
    bool energy_compensation,
    out BSDF bsdf
)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(weight);
    mx_stack.set_bsdf(bsdf.get_index(), create_mx_oren_nayar(color, roughness, normal, energy_compensation));
    mx_prepare_bsdf_with_default_tangent(bsdf, normal);
}

void mx_burley_diffuse_bsdf(
    ClosureData closure_data,
    float weight,
    float3 color,
    float roughness,
    float3 normal,
    out BSDF bsdf
)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(weight);
    mx_stack.set_bsdf(bsdf.get_index(), create_mx_burley(color, roughness, normal));
    mx_prepare_bsdf_with_default_tangent(bsdf, normal);
}

void mx_translucent_bsdf(ClosureData closure_data, float weight, float3 color, float3 normal, out BSDF bsdf)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(weight);
    mx_stack.set_bsdf(bsdf.get_index(), create_mx_translucent(color, normal));
    mx_prepare_bsdf_with_default_tangent(bsdf, normal);
}

void mx_subsurface_bsdf(
    ClosureData closure_data,
    float weight,
    float3 color,
    float3 radius,
    float anisotropy,
    float3 normal,
    out BSDF bsdf
)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(weight);
    // MX139 maps MaterialX subsurface to a diffuse reflection fallback.
    mx_stack.set_bsdf(bsdf.get_index(), create_mx_subsurface_diffuse_fallback(color));
    mx_prepare_subsurface_diffuse_fallback_bsdf(bsdf, normal);
}

void mx_beer_bsdf(
    ClosureData closure_data,
    float weight,
    float3 absorption,
    float pathshortening,
    float3 normal,
    out BSDF bsdf
)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(weight);
    mx_stack.set_bsdf(bsdf.get_index(), create_mx_beer(absorption, pathshortening));
    mx_prepare_bsdf_with_default_tangent(bsdf, normal);
}

void mx_scratchconductor_brdf(
    ClosureData closure_data,
    float weight,
    float3 ior,
    float3 extinction,
    float2 roughness,
    float3 scratchdirection,
    float depth,
    float3 mask,
    float scratchRoughness,
    float3 normal,
    float3 tangent,
    int distribution,
    out BSDF bsdf
)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(weight);
    mx_stack.set_bsdf(
        bsdf.get_index(),
        create_mx_scratch_conductor(
            ior,
            extinction,
            roughness,
            float3(scratchdirection[1], scratchdirection[0], scratchdirection[2]) * 2.0 - 1.0,
            max(mask[1] * depth, 0.0001),
            float3(clamp(mask[0], 0.0, 1.0), mask[1], mask[2]),
            clamp(scratchRoughness, 0.0, 1.0),
            normal,
            tangent,
            distribution
        )
    );
    mx_prepare_bsdf(bsdf, normal, si.shading_frame_ws.tangent);
}

void mx_dielectric_bsdf(
    ClosureData closure_data,
    float weight,
    float3 tint,
    float ior,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution,
    int scatter_mode,
    out BSDF bsdf
)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(weight);
    mx_stack.set_bsdf(
        bsdf.get_index(),
        create_mx_dielectric(
            tint,
            ior,
            roughness,
            retroreflective,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
            distribution,
            scatter_mode,
            si.front_facing
        )
    );
    mx_prepare_bsdf(bsdf, normal, tangent);
}

void mx_conductor_bsdf(
    ClosureData closure_data,
    float weight,
    float3 ior,
    float3 extinction,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution,
    out BSDF bsdf
)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(weight);
    mx_stack.set_bsdf(
        bsdf.get_index(),
        create_mx_conductor(
            ior,
            extinction,
            roughness,
            retroreflective,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
            distribution
        )
    );
    mx_prepare_bsdf(bsdf, normal, tangent);
}

void mx_generalized_schlick_bsdf(
    ClosureData closure_data,
    float weight,
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
    float2 roughness,
    bool retroreflective,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 normal,
    float3 tangent,
    int distribution,
    int scatter_mode,
    out BSDF bsdf
)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(weight);
    mx_stack.set_bsdf(
        bsdf.get_index(),
        create_mx_generalized_schlick(
            color0,
            color82,
            color90,
            exponent,
            roughness,
            retroreflective,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
            distribution,
            scatter_mode,
            si.front_facing
        )
    );
    mx_prepare_bsdf(bsdf, normal, tangent);
}

void mx_sheen_bsdf(
    ClosureData closure_data,
    float weight,
    float3 color,
    float roughness,
    float3 normal,
    int mode,
    out BSDF bsdf
)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(weight);
    mx_stack.set_bsdf(bsdf.get_index(), create_mx_sheen(color, roughness, normal, mode, si.front_facing));
    mx_prepare_bsdf(bsdf, normal, si.shading_frame_ws.tangent);
}

void mx_chiang_hair_bsdf(
    ClosureData closure_data,
    float3 tint_R,
    float3 tint_TT,
    float3 tint_TRT,
    float ior,
    float2 roughness_R,
    float2 roughness_TT,
    float2 roughness_TRT,
    float cuticle_angle,
    float3 absorption_coefficient,
    float3 normal,
    float3 curve_direction,
    out BSDF bsdf
)
{
    bsdf.set_bsdf(mx_stack_next_bsdf++);
    mx_stack.bsdf_weights[bsdf.get_index()] = float3(1.0);
    mx_stack.set_bsdf(
        bsdf.get_index(),
        create_mx_chiang_hair(
            tint_R,
            tint_TT,
            tint_TRT,
            ior,
            roughness_R,
            roughness_TT,
            roughness_TRT,
            cuticle_angle,
            absorption_coefficient,
            normal,
            curve_direction
        )
    );
    mx_prepare_curve_bsdf(bsdf, normal, curve_direction);
}
