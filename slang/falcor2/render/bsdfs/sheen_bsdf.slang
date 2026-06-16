// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Sheen-only implementation from Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF";
/// all energy which is not reflected will be transmitted as a Dirac impulse
public struct SheenBSDF : IBSDF
{
    /// Sheen reflectivity.
    public float3 color = {};
    /// Surface roughness (max 1.0).
    public float roughness = {};

    public __init() { }

    public __init(float3 color, float roughness)
    {
        this.color = color;
        this.roughness = roughness;
    }

    float sheen_D(float n_dot_h)
    {
        float inv_alpha = 1.0 / roughness;
        float cos2h = n_dot_h * n_dot_h;
        float sin2h = max(1.0 - cos2h, 0.0078125);
        return (2.0 + inv_alpha) * pow(sin2h, inv_alpha * 0.5) / (2.0 * float.PI);
    }

    float L(float x)
    {
        const float r = (1.0 - roughness) * (1.0 - roughness);
        const float one_min_r = 1.0 - r;

        const float a = 25.3245 * r + 21.5473 * one_min_r;
        const float b = 3.32435 * r + 3.82987 * one_min_r;
        const float c = 0.16801 * r + 0.19823 * one_min_r;
        const float d = (-1.27393) * r + (-1.97760) * one_min_r;
        const float e = (-4.85967) * r + (-4.32054) * one_min_r;

        return a / (1.0 + b * pow(x, c)) + d * x + e;
    }

    float lambda(float cos)
    {
        if (cos < 0.5)
            return exp(L(cos));
        else
            return exp(2.0 * L(0.5) - L(1.0 - cos));
    }

    float G(float n_dot_o, float n_dot_i)
    {
        // this is the full visibility function; note that we have the questionable terminator softening from equ. 4
        return 1.0 / (1.0 + pow(lambda(n_dot_o), 1.0 + 2.0 * pow(1.0 - n_dot_o, 8.0)) + lambda(n_dot_i));
    }

    public float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, const BSDFContext bc)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return float3(0.0);

        float3 wh = normalize(wi + wo);
        float n_dot_h = wh.z;

        const float3 D = sheen_D(n_dot_h);
        // We multiply by wo.z
        float3 Fr = D * G(wo.z, wi.z) * color / (4.0 * wi.z);

        return Fr;
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
        // Default initialization to avoid divergence at returns.
        wo = {};
        weight = {};
        pdf = {};
        lobe_types = {};

        // Without a baselayer, this BSDF should be transparent. Until we can overwrite local importance sampling with
        // the LobeTypess, we will pretend for sample() that there is no transparency. This is a performance
        // optimisation for the LayeredMaterialInstance.
        // TODO: restore transparent sampling once local importance sampling can account for lobe types.
        /*
        float3 albedo_t = eval_albedo(wi).transmission;
        const float selection_weight = luminance(albedo_t);
        if (next_float(sg) < selection_weight)
        {
            wo = float3(-wi.x, -wi.y, -wi.z);
            pdf = 0.0;
            weight = 1.0;
            lobe_types = LobeTypes::delta_transmission;
            return true;
        }*/

        // The paper recommends cosine sampling.
        wo = square_to_cosine_hemisphere_concentric(next_float2(sg));
        pdf = square_to_cosine_hemisphere_concentric_pdf(wo);

        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            weight = {};
            return false;
        }

        lobe_types = LobeTypes::diffuse_reflection;

        weight = (eval(wi, wo, sg, bc) / (float.INV_PI* wo.z));
        return true;
    }

    public float eval_pdf(const float3 wi, const float3 wo, BSDFContext bc)
    {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA)
            return 0.0;

        float3 albedo_t = eval_albedo(wi).transmission;
        return float.INV_PI * wo.z;
    }

    public AlbedoContributions eval_albedo(const float3 wi)
    {
        const float coefficients[11][5]
            = { { 1.01108, 75.2286, -9.4671, 0.000973445, -1.01325 },
                { 8.44447, 0.0135054, 1.11637, 0.479139, -9.30873 },
                { 5.67124, 0.0194668, 1.12489, 0.453551, -6.53104 },
                { 4.50086, 0.0237758, 1.09376, 0.423665, -5.36521 },
                { 3.77781, 0.0273286, 1.0666, 0.400267, -4.65601 },
                { 3.35338, 0.0307392, 0.989189, 0.371074, -4.24089 },
                { 3.39605, 0.0308828, 0.845455, 0.333813, -4.28662 },
                { 3.62583, 0.0303531, 0.79754, 0.325588, -4.51725 },
                { 4.11524, 0.0172797, 0.701166, 0.275242, -5.04394 },
                { 4.4012, 0.0159469, 0.685892, 0.270221, -5.32997 },
                { 5.8912, 0.0171461, 0.66366, 0.288735, -6.78867 } };

        // fit was [1, 21] for cosI(0) = 21
        const float x = (wi.z) * 20.0 + 1.0;

        // roughness precomputed between [0,1] in 0.1 steps
        const int index_b = max(int(floor(roughness * 10)), 0);
        const int index_t = min(index_b + 1, 10);
        const float index_int = index_t - (roughness * 10.0);

        // albedo = e + pow(x, d) + a/(1.0 + b * pow(x,c))
        const float albedo_tb = coefficients[index_b][4] + pow(x, coefficients[index_b][3])
            + coefficients[index_b][0] / (1.0 + coefficients[index_b][1] * pow(x, coefficients[index_b][2]));
        const float albedo_tt = coefficients[index_t][4] + pow(x, coefficients[index_t][3])
            + coefficients[index_t][0] / (1.0 + coefficients[index_t][1] * pow(x, coefficients[index_t][2]));
        const float albedo_r = 1.0 - clamp(albedo_tb * index_int + albedo_tt * (1.0 - index_int), 0.0, 1.0);

        return AlbedoContributions(color * albedo_r, albedo_r - (albedo_r * color), (1.0 - albedo_r), 0.0);
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        return RoughnessInformation(float2(0.5), float3(0.0));
    }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(1.0);
        }
    }
};
