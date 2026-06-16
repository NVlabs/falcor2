// SPDX-License-Identifier: Apache-2.0
//
// Public Falcor-facing contract for active Mx139 BSDFs.
//
// Mx139 BSDF math is derived from BSDL/OSL sources, but active Slang code uses
// Falcor direction names throughout: `wi` is the incident/view direction and
// `wo` is the scattered/output direction. Source comments may point to BSDL
// formulas; in those cases the variables have already been translated to this
// convention.

implementing mx139;

import falcor2.utils;
import falcor2.render;

namespace mx139 {

// OSL-derived MX139 generated graphs call sample_texture() for MaterialX
// image nodes. Keeping the wrapper in the profile module makes generated
// shaders smaller while preserving base-LOD sampling behavior through the
// supplied lod sampler.
public float4 sample_texture<TLodSampler : ILodSampler>(
    TextureHandle handle,
    float2 uv,
    TLodSampler lod_sampler,
    float4 default_value = {}
)
{
    return handle.sample(lod_sampler, uv, default_value);
}

public interface ILayerableBSDF : IBSDF
{
    /// Directional scattering albedo for an incident/view direction in the
    /// local Falcor shading frame.
    public float3 directional_albedo(const float3 wi);

    /// Upper-layer opacity for an incident/view direction in the local Falcor
    /// shading frame. `1 - layer_opacity(wi)` is the share available to a
    /// lower MaterialX layer.
    public float3 layer_opacity(const float3 wi);
};

/// Internal BSDF sample primitive shared by Mx139 BSDF math.
///
/// `weight` is the BSDL-style eval/pdf sample weight. The bridge to Falcor
/// `IBSDF::eval()` multiplies it by `pdf` to return Falcor's f*cos value.
public struct MxBsdfSample
{
    public float3 wo;
    public float3 weight;
    public float pdf;
    public float roughness;

    public __init()
    {
        wo = float3(0.0);
        weight = float3(0.0);
        pdf = 0.0;
        roughness = 0.0;
    }

    public __init(float3 wo, float3 weight, float pdf, float roughness)
    {
        this.wo = wo;
        this.weight = weight;
        this.pdf = pdf;
        this.roughness = roughness;
    }

    [mutating]
    public void update(float3 ow, float opdf, float cpdf)
    {
        if (cpdf > PDF_MIN) {
            opdf *= cpdf;
            ow *= 1.0 / cpdf;
            if (opdf == pdf)
                weight = lerp(weight, ow, 0.5);
            else if (opdf < pdf)
                weight = lerp(ow, weight, 1.0 / (1.0 + opdf / pdf));
            else
                weight = lerp(weight, ow, 1.0 / (1.0 + pdf / opdf));
            pdf += opdf;
        }
    }
};

public MxBsdfSample
eval_diffuse_reflection_sample(const float3 wi, const float3 wo, const float3 weight, const float roughness)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return MxBsdfSample();
    return MxBsdfSample(wo, weight, wo.z * float.INV_PI, roughness);
}

} // namespace mx139
