// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing scene;

public static const int UV_COUNT = 2;

/// Filter modes for texture sampling
public enum TextureFilterMode : uint32_t {
    Nearest = 0,
    Linear = 1,
};

/// Wrap modes for texture sampling
public enum TextureWrapMode : uint32_t {
    Repeat = 0,
    MirroredRepeat = 1,
    ClampToEdge = 2,
};

/// Alpha modes for materials
public enum AlphaMode : uint32_t {
    /// The alpha value is ignored and the rendered output is fully opaque
    Opaque = 0,
    /// The rendered output is either fully opaque or fully transparent depending on the alpha value and the specified
    /// alpha cutoff value
    Mask = 1,
    /// The alpha value is used to composite the source and destination areas
    Blend = 2,
};

/// Geometry ID.
/// Indexes into Scene::geometries.
public enum class GeometryID : uint { invalid = 0xffffffff };

/// Geometry instance ID.
/// Indexes into Scene::geometry_instances.
/// Corresponds to InstanceID() + GeometryIndex() system values.
public enum class GeometryInstanceID : uint { invalid = 0xffffffff };

/// Material ID.
/// Indexes into Scene::materials.
public enum class MaterialID : uint { invalid = 0xffffffff };

/// Texture ID.
/// Indexes into Scene::textures.
public enum class TextureID : uint { invalid = 0xffffffff };

public struct Vertex
{
    public float3 position;
    public float3 normal;
    public float3 tangent;
    public float3 bitangent;
    public float3 color;
    public float2 uv[UV_COUNT];
}

public struct VertexData
{
    public float3 pos_ws;
    public float3 normal_ws;
    public float3 tangent_ws;
    public float3 bitangent_ws;
    public float3 face_normal_ws;
    public float3 color;
    public float2 uv[UV_COUNT];
}

// Unpacked material
public struct MaterialTexture
{
    public TextureID texture;
    public float3x3 transform;
    public uint32_t tex_coord;
    public TextureWrapMode wrap_s;
    public TextureWrapMode wrap_t;
    public TextureFilterMode filter;
}

// Type can be used to handle different material models
// Currently it's just used to handle conversion between spec/gloss vs metal/rough
// on the fly.
// For a metal/rough material:
//    type==0, albedo==albedo, specular=F0, roughness=roughness, metallic=metallic
// For a spec/gloss material:
//    type==1, albedo==diffuse, specular=specular, roughness=glossiness
public struct MaterialData
{
    public int type;
    public MaterialTexture albedo;
    public MaterialTexture emission;
    public MaterialTexture normal;
    public MaterialTexture roughness;
    public MaterialTexture metallic;
    public MaterialTexture specular;
    public float normal_map_strength;
    public float emission_strength;
    public bool double_sided;
    public AlphaMode alpha_mode;
    public float alpha_cutoff;
}

// Unpacked material
public struct Material : IDifferentiable
{
    public float3 albedo;
    public float3 emission;
    public float3 normal;
    public float3 specular;
    public float roughness;
    public float metallic;
    public float alpha;
}

public struct GeometryInstance
{
    public uint flags;
    public MaterialID material;
    public uint vertex_buffer;
    public uint vertex_start;
    public uint vertex_count;
    public uint index_buffer;
    public uint index_start;
    public uint index_count;
}

public struct Camera
{
    public float3 position;
    public float3 image_u;
    public float3 image_v;
    public float3 image_w;
    public int2 dims;

    public Ray get_ray(float2 uv)
    {
        uv = uv * 2 - 1;
        float3 dir = normalize(uv.x * image_u + uv.y * image_v + image_w);
        return Ray(position, dir);
    }
}

// Solves the quadratic equation to find the metallic value given diffuse/specular
static const float F0 = 0.04f;
static const float DSG_EPSILON = 1e-3f;
float solve_metallic(float diffuse, float specular, float one_minus_spec_strength)
{
    if (specular < F0) {
        return 0;
    }
    var a = F0;
    var b = diffuse * one_minus_spec_strength / (1 - F0) + specular - 2 * F0;
    var c = F0 - specular;
    var D = b * b - 4 * a * c;
    return clamp((-b + sqrt(D)) / (2 * a), 0, 1);
}

// Convert a specular/glossiness material to an albedo/metallic/roughness material
// Based on
// https://kcoley.github.io/glTF/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows-bjs/js/babylon.pbrUtilities.js
// with some tweaks.
void dsg_to_amr(
    float3 diffuse,
    float3 specular,
    float glossiness,
    out float3 out_albedo,
    out float out_metallic,
    out float out_roughness
)
{

    // Solve for metallic
    var one_minus_spec_strength = 1 - max(max(specular.x, specular.y), specular.z);
    var metallic = solve_metallic(luminance(diffuse), luminance(specular), one_minus_spec_strength);

    // Calculate albedo in both forms, so we can take whichever one doesn't end up at infinity
    var albedo_from_diffuse = diffuse / (1 - min(metallic, 1 - DSG_EPSILON));
    var albedo_from_specular = (specular - F0 * (1 - metallic)) / max(metallic, DSG_EPSILON);
    var albedo = lerp(albedo_from_diffuse, albedo_from_specular, metallic * metallic);

    // Clamp and output
    out_albedo = clamp(albedo, 0, 1);
    out_metallic = metallic;
    out_roughness = clamp(1 - glossiness, 0.05f, 1.0f);
}
