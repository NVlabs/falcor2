// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// MaterialX sheen BSDF ported from OSL/BSDL MTX.
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_sheen_impl.h
// BSDL: external/dbg_OSL/src/libbsdl/include/BSDL/MTX/bsdf_sheen_decl.h
//
// Implements:
//   mode == 0: ContyKulla MTX (non-legacy shadowing) - bsdf_sheen_impl.h:90-202
//   mode == 1: Zeltner-Burley LTC (LTC_SAMPLING=true mode) - bsdf_sheen_impl.h:204-353

implementing mx139;

namespace mx139 {

// BSDL: bsdf_sheen_impl.h:90-150 ContyKullaDist<false> + SheenMicrofacet::eval/sample
// + bsdf_sheen_impl.h:189-202 ContyKullaSheenMTX::albedo rational fit.
public struct MxContyKullaSheenMtx
{
    public float roughness;

    public __init(float rough) { roughness = max(rough, 0.06); }

    public float D(float3 Hr)
    {
        float cos_theta = Hr.z;
        float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        return pow(sin_theta, 1.0 / roughness) * (2.0 + 1.0 / roughness) * float.INV_TWO_PI;
    }

    public float G2(float3 wo, float3 wi)
    {
        const float cos_NI = wi.z;
        const float cos_NO = wo.z;
        return (cos_NI * cos_NO) / (cos_NI + cos_NO - cos_NI * cos_NO);
    }

    public MxBsdfSample eval(float3 wo, float3 wi)
    {
        if (wi.z <= 1e-5 || wo.z <= 1e-5)
            return MxBsdfSample();
        const float Dv = D(normalize(wo + wi));
        if (Dv < 1e-6)
            return MxBsdfSample();
        const float G2v = G2(wo, wi);
        return MxBsdfSample(wi, float3(Dv * G2v * float.PI_OVER_TWO / wo.z), float.INV_TWO_PI, 0.0);
    }

    public MxBsdfSample sample(float3 wo, float3 rnd)
    {
        return eval(wo, square_to_uniform_hemisphere_concentric(rnd.xy));
    }

    public float albedo(float cos_NO)
    {
        const float x = cos_NO;
        const float y = roughness;
        const float2 r = float2(13.67300, 1.0) + float2(-68.78018, 61.57746) * x + float2(799.08825, 442.78211) * y
            + float2(-905.00061, 2597.49308) * x * y + float2(60.28956, 121.81241) * x * x
            + float2(1086.96473, 3045.55075) * y * y;
        return clamp(r.x / r.y, 0.0, 1.0);
    }
};

// BSDL: bsdf_sheen_impl.h:204-353 ZeltnerBurleySheen (LTC_SAMPLING=true mode).
public struct MxZeltnerBurleySheen
{
    public float roughness;

    public __init(float rough) { roughness = clamp(max(rough, 0.02), 0.0, 1.0); }

    public float3 fetch_coeffs(float cos_NO)
    {
        float row = clamp(roughness, 0.0, ALMOST_ONE) * float(ZELTNER_SHEEN_LTC_RES - 1);
        float col = clamp(cos_NO, 0.0, ALMOST_ONE) * float(ZELTNER_SHEEN_LTC_RES - 1);
        int ri = int(floor(row));
        int ci = int(floor(col));
        float rf = row - float(ri);
        float cf = col - float(ci);
        float3 v1 = zeltner_sheen_ltc_param(uint(ri), uint(ci));
        float3 v2 = zeltner_sheen_ltc_param(uint(ri), uint(ci + 1));
        float3 v3 = zeltner_sheen_ltc_param(uint(ri + 1), uint(ci));
        float3 v4 = zeltner_sheen_ltc_param(uint(ri + 1), uint(ci + 1));
        return lerp(lerp(v1, v2, cf), lerp(v3, v4, cf), rf);
    }

    public MxBsdfSample eval(float3 wo, float3 wi)
    {
        if (wo.z < 0.0 || wi.z <= 0.0)
            return MxBsdfSample();
        return eval(wo, wi, fetch_coeffs(wo.z));
    }

    public MxBsdfSample eval(float3 wo, float3 wi, float3 ltc)
    {
        const float a_inv = ltc.x;
        const float b_inv = ltc.y;
        const float r_coeff = ltc.z;
        const float3 wi_orig = float3(a_inv * wi.x + b_inv * wi.z, a_inv * wi.y, wi.z);
        const float len2 = dot(wi_orig, wi_orig);
        const float jacobian = (a_inv / len2) * (a_inv / len2);
        const float pdf = jacobian * max(wi_orig.z, 0.0) * float.INV_PI;
        if (pdf > float.MIN)
            return MxBsdfSample(wi, float3(r_coeff), pdf, roughness);
        return MxBsdfSample();
    }

    public MxBsdfSample sample(float3 wo, float3 rnd)
    {
        if (wo.z < 0.0)
            return MxBsdfSample();
        const float3 ltc = fetch_coeffs(wo.z);
        const float a_inv = ltc.x;
        const float b_inv = ltc.y;
        const float3 wi_orig = square_to_cosine_hemisphere_concentric(rnd.xy);
        const float3 wi = float3(wi_orig.x - wi_orig.z * b_inv, wi_orig.y, a_inv * wi_orig.z);
        return eval(wo, normalize(wi), ltc);
    }

    public float albedo(float cos_NO) { return fetch_coeffs(cos_NO).z; }
};

public struct MxSheenBSDF : ILayerableBSDF
{
    // FRAME CONVENTION WARNING
    //
    // Sheen is different from the other mx139 BSDFs.
    //
    // Most BSDFs are strictly material-local: the caller converts world
    // directions into material/local space once, then the BSDF uses those
    // directions directly. OSL/BSDL sheen has one extra frame step that is
    // part of the sheen lobe itself, not part of external MaterialInstance
    // world/local handling.
    //
    // In BSDL, SheenLobe derives from Lobe with:
    //   Z = visible_normal(data.N)
    //   X = globals.wo
    // so the internal sheen microfacet/LTC evaluation is view-oriented. The
    // BSDL's wrapper then calls Base::frame.local(wo/wi), meaning even in a
    // material-local case where N=(0,0,1), sheen still evaluates in a
    // secondary frame whose X axis is the outgoing direction.
    //
    // This implementation mirrors that behavior with:
    //   Frame(float3(0,0,1), wi)
    // inside eval() and sample(). Do not remove this when adding the later
    // MaterialInstance/frame layer. The future external frame layer should only
    // convert world directions into the material-local convention before
    // calling the Falcor-facing ILayerableBSDF BSDF. Sheen must still create
    // this internal view frame.
    //
    // Consequence for tests: the OSL probe uses direct eval_impl/sample_impl
    // for most leafs, but NOT for sheen. Sheen parity intentionally goes
    // through the BSDL wrapper path because that wrapper frame is part of the
    // BSDL sheen BSDF behavior we are cloning.
    public float3 tint = {};
    public float base_roughness = {};
    public float sheen_alpha = {};
    public float Emiss = {};
    public int sheen_mode = {};
    public bool backfacing = {};

    public __init() { }

    // BSDL: bsdf_sheen_impl.h:19-48 SheenLobe constructor
    public __init(float3 albedo_value, float roughness, int mode, bool backfacing)
    {
        tint = albedo_value;
        base_roughness = clamp(roughness, 0.0, 1.0);
        sheen_mode = mode;
        this.backfacing = backfacing;
        const bool zeltner = use_zeltner();
        sheen_alpha = zeltner ? max(0.02, sqrt(base_roughness)) : max(0.06, base_roughness);
        if (backfacing) {
            Emiss = 1.0;
        } else if (zeltner) {
            MxZeltnerBurleySheen sheen = MxZeltnerBurleySheen(sheen_alpha);
            Emiss = 1.0 - min(sheen.albedo(1.0) * mx_hmax(tint), 1.0);
        } else {
            MxContyKullaSheenMtx sheen = MxContyKullaSheenMtx(sheen_alpha);
            Emiss = 1.0 - min(sheen.albedo(1.0) * mx_hmax(tint), 1.0);
        }
    }

    // ILayerableBSDF

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        MxBsdfSample r = eval_bsdf_sample(wi, wo);
        return r.pdf > 0.0 ? r.weight * r.pdf : float3(0.0);
    }

    public bool sample<S : ISampleGenerator>(
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 weight,
        out LobeTypes lobe_types,
        inout S sg,
        const BSDFContext bc
    )
    {
        MxBsdfSample r = sample_bsdf_sample(wi, next_float3(sg));
        if (r.pdf <= 0.0) {
            wo = float3(0.0);
            pdf = 0.0;
            weight = float3(0.0);
            lobe_types = LobeTypes::none;
            return false;
        }
        wo = r.wo;
        pdf = r.pdf;
        weight = r.weight;
        lobe_types = LobeTypes::glossy_reflection;
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, const BSDFContext bc)
    {
        return eval_bsdf_sample(wi, wo).pdf;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        // Sheen is reflection-only, but MaterialX 1.38 layering expects
        // eval_albedo().transmission to carry the BSDL layer pass-through.
        const float E_ms = layer_energy(wi);
        float3 reflection = saturate(tint * (1.0 - E_ms));
        float3 transmission = saturate(float3(E_ms));
        return AlbedoContributions(
            reflection,
            max(float3(0.0), float3(1.0) - reflection - transmission),
            transmission,
            float3(0.0)
        );
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(base_roughness, base_roughness), float3(0.0));
    }

    public float3 directional_albedo(const float3 wi) { return tint * (1.0 - layer_energy(wi)); }

    public float3 layer_opacity(const float3 wi) { return saturate(float3(1.0) - float3(layer_energy(wi))); }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }

    // Implementation

    private bool use_zeltner() { return sheen_mode == 1; }

    private float layer_energy(const float3 wi)
    {
        if (backfacing)
            return 1.0;
        const float cos_no = max(wi.z, 0.0);
        if (cos_no >= 1.0)
            return Emiss;
        if (use_zeltner()) {
            MxZeltnerBurleySheen sheen = MxZeltnerBurleySheen(sheen_alpha);
            return 1.0 - min(sheen.albedo(cos_no) * mx_hmax(tint), 1.0);
        }
        MxContyKullaSheenMtx sheen = MxContyKullaSheenMtx(sheen_alpha);
        return 1.0 - min(sheen.albedo(cos_no) * mx_hmax(tint), 1.0);
    }

    // BSDL: bsdf_sheen_impl.h:50-67 SheenLobe::eval_impl
    private MxBsdfSample eval_bsdf_sample(const float3 wi, const float3 wo)
    {
        const bool do_self = wo.z > 0.0 && wi.z >= 0.0 && !backfacing;
        if (!do_self)
            return MxBsdfSample();
        Frame frame = Frame(float3(0.0, 0.0, 1.0), wi);
        const float3 local_wi = frame.to_local(wi);
        const float3 local_wo = frame.to_local(wo);
        MxBsdfSample s = use_zeltner() ? MxZeltnerBurleySheen(sheen_alpha).eval(local_wi, local_wo)
                                       : MxContyKullaSheenMtx(sheen_alpha).eval(local_wi, local_wo);
        s.wo = wo;
        s.weight *= tint;
        s.roughness = base_roughness;
        return s;
    }

    // BSDL: bsdf_sheen_impl.h:69-84 SheenLobe::sample_impl
    private MxBsdfSample sample_bsdf_sample(const float3 wi, const float3 rnd)
    {
        if (backfacing)
            return MxBsdfSample();
        Frame frame = Frame(float3(0.0, 0.0, 1.0), wi);
        const float3 local_wi = frame.to_local(wi);
        MxBsdfSample s = use_zeltner() ? MxZeltnerBurleySheen(sheen_alpha).sample(local_wi, rnd)
                                       : MxContyKullaSheenMtx(sheen_alpha).sample(local_wi, rnd);
        s.wo = frame.to_global(s.wo);
        s.weight *= tint;
        s.roughness = base_roughness;
        return s;
    }
};

public MxSheenBSDF create_mx_sheen(float3 color, float roughness, float3 normal, int mode, bool front_facing)
{
    // Mx139 follows the authored MaterialX mode.
    return MxSheenBSDF(color, roughness, mode, !front_facing);
}

public MxSheenBSDF create_mx_sheen(float3 color, float roughness, float3 normal, int mode)
{
    return create_mx_sheen(color, roughness, normal, mode, true);
}

} // namespace mx139
