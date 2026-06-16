// SPDX-License-Identifier: Apache-2.0

void mx_surface_unlit(
    float emission,
    float3 emission_color,
    float transmission,
    float3 transmission_color,
    float opacity,
    out surfaceshader result
)
{
    float trans = clamp(transmission, 0.0, 1.0);
    result.root = {};
    if (trans > 0.0) {
        BSDF transparent_bsdf = {};
        transparent_bsdf.set_bsdf(mx_stack_next_bsdf++);
        int transparent_index = transparent_bsdf.get_index();
        mx_stack.bsdf_weights[transparent_index] = float3(1.0);
        mx_stack.bsdf_transmission_scales[transparent_index] = float3(1.0);
        mx_stack.bsdf_n[transparent_index] = float3(0.0, 0.0, 1.0);
        mx_stack.bsdf_t[transparent_index] = float3(1.0, 0.0, 0.0);
#ifdef MX139_STACK_HAS_BSDF_FRAMES
        // bsdf_mix layering samples through stored frames rather than n/t pairs.
        mx_stack.bsdf_frames[transparent_index]
            = Frame(mx_stack.bsdf_n[transparent_index], mx_stack.bsdf_t[transparent_index]);
#endif
        mx_stack.set_bsdf(transparent_index, create_mx_transparent(trans * transmission_color));
        result.root = transparent_bsdf;
    }
    result.opacity = clamp(opacity, 0.0, 1.0);
    // Match mx_surfacematerial/OSL behavior: surface opacity gates emission as
    // well as scattering. The material wrapper reads edf.radiance directly.
    result.edf.radiance = result.opacity * (1.0 - trans) * emission * emission_color;
    result.thin_walled = false;
}
