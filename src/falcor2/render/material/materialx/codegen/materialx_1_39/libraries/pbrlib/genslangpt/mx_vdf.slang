// SPDX-License-Identifier: Apache-2.0

void mx_absorption_vdf(ClosureData closure_data, float3 absorption, out VDF vdf)
{
    // TODO(@tdavidovic): This is raster-compatible VDF emulation. MX139 does
    // not have full volume scattering yet, so keep only Beer-style attenuation.
    vdf.throughput = exp(-max(absorption, float3(0.0)));
}

void mx_anisotropic_vdf(ClosureData closure_data, float3 absorption, float3 scattering, float anisotropy, out VDF vdf)
{
    // TODO(@tdavidovic): This intentionally ignores scattering and anisotropy
    // until MX139 grows real VDF support.
    mx_absorption_vdf(closure_data, absorption, vdf);
}

void mx_apply_vdf_transmission(BSDF bsdf, VDF vdf)
{
    if (bsdf.is_bsdf())
        mx_stack.bsdf_transmission_scales[bsdf.get_index()] *= vdf.throughput;
    else
        mx_stack.layer_transmission_scales[bsdf.get_index()] *= vdf.throughput;
}

void mx_layer_vdf(ClosureData closure_data, BSDF top, VDF base, out BSDF result)
{
    // TODO(@tdavidovic): Raster treats VDF layering as transmission
    // throughput. This mirrors that compatibility behavior without full VDF
    // transport.
    mx_apply_vdf_transmission(top, base);
    result = top;
}
