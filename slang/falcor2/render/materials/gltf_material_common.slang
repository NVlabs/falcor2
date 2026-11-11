// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

// The handle and packed selector are kept together logically even when a
// material chooses a physically flat layout after backend performance tests.
public struct GLTFTextureInfo
{
    public TextureHandle handle;
    public uint packed;

    public uint texcoord() { return packed & gltfmat::TextureInfoBits::TEXCOORD_MASK; }
    public bool normal_map_encoded() { return (packed & gltfmat::TextureInfoBits::NORMAL_MAP_ENCODED) != 0u; }
};

public static const uint GLTF_FEATURE_RECORD_SIZE = 32u;
public static const uint GLTF_TRANSFORM_RECORD_SIZE = 24u;

// Each active extension consumes one 32-byte slot in ascending FeatureBits
// order. The padding fields are part of the ABI and are zeroed by the writer.
// The scalar component mapping is fixed across C++ and Slang.
public struct GLTFSpecularRecord
{
    public GLTFTextureInfo specular;
    public GLTFTextureInfo specular_color;
    public float4 color_strength; // RGB color factor, A strength factor.
};

public struct GLTFTransmissionRecord
{
    public GLTFTextureInfo texture;
    public uint2 reserved;
    public float4 factor_and_reserved; // X transmission factor.
};

public struct GLTFIORRecord
{
    public float4 ior_and_reserved; // X IOR, including the glTF zero sentinel.
    public float4 reserved;
};

public struct GLTFVolumeRecord
{
    public float4 sigma_a; // XYZ Beer-Lambert absorption coefficients.
    public float4 reserved;
};

public struct GLTFClearcoatRecord
{
    public GLTFTextureInfo texture;
    public GLTFTextureInfo roughness_texture;
    public GLTFTextureInfo normal_texture;
    public float16_t4 factors; // X coat, Y roughness, Z normal scale.
};

public struct GLTFSheenRecord
{
    public GLTFTextureInfo color_texture;
    public GLTFTextureInfo roughness_texture;
    public float4 color_roughness; // RGB color, A roughness.
};

public struct GLTFIridescenceRecord
{
    public GLTFTextureInfo texture;
    public GLTFTextureInfo thickness_texture;
    public float4 factor_ior_thickness; // Factor, IOR, minimum, maximum.
};

public struct GLTFAnisotropyRecord
{
    public GLTFTextureInfo texture;
    public uint2 reserved;
    public float4 strength_rotation; // X strength, Y angle in radians.
};

public struct GLTFDiffuseTransmissionRecord
{
    public GLTFTextureInfo texture;
    public GLTFTextureInfo color_texture;
    public float4 color_factor; // RGB color factor, A transmission factor.
};

public struct GLTFTextureUVTransform
{
    public float3 row_u;
    public float3 row_v;
};

public struct GLTFVolumeData
{
    public float3 sigma_a;
};

public struct GLTFSheenData
{
    public float3 color;
    public float roughness;
    public TextureHandle color_texture;
    public TextureHandle roughness_texture;
    public uint selectors;
};

public struct GLTFIridescenceData
{
    public float factor;
    public float ior;
    public float thickness_minimum;
    public float thickness_maximum;
    public TextureHandle texture;
    public TextureHandle thickness_texture;
    public uint selectors;
};

// Optional record immediately preceding sparse texture transforms.
// Buffer layout size: sizeof includes expanded resource handles on some targets.
public static const uint GLTF_SURFACE_EXTENSIONS_SIZE = 40;
public struct GLTFSurfaceExtensionsData
{
    public float3 diffuse_transmission_color_factor;
    public float anisotropy_strength;
    public float anisotropy_rotation;
    public float diffuse_transmission_factor;
    public TextureHandle anisotropy_texture;
    public TextureHandle diffuse_transmission_texture;
    public TextureHandle diffuse_transmission_color_texture;
    public uint selectors;
};

public struct GLTFClearcoatData
{
    public TextureHandle texture;
    public TextureHandle roughness_texture;
    public TextureHandle normal_texture;
    public float normal_scale;
    public uint selectors;
};

namespace gltf_material {
// Offsets are relative to each material's extra_data reference. Callers guard
// loads with nested compile-time and runtime bit tests.
// MULTISCATTER is deliberately outside EXTENSION_BITS and allocates no record.
public uint feature_record_offset(uint feature_mask, uint bit)
{
    const uint extensions = feature_mask & gltfmat::FeatureBits::EXTENSION_BITS;
    return GLTF_FEATURE_RECORD_SIZE * countbits(extensions & (bit - 1u));
}

public uint transform_record_offset(uint feature_mask, uint transform_mask, uint slot_bit)
{
    const uint extensions = feature_mask & gltfmat::FeatureBits::EXTENSION_BITS;
    return GLTF_FEATURE_RECORD_SIZE * countbits(extensions)
        + GLTF_TRANSFORM_RECORD_SIZE * countbits(transform_mask & (slot_bit - 1u));
}

// SlotBit is a generic value so disabled texture, transform, and UV-set paths
// can be removed when a material is instantiated with exact compile masks.
public float2 transform_uv_sparse<let Transforms : uint, let SlotBit : uint>(
    float2 uv,
    uint feature_mask,
    uint transform_mask,
    BufferPtr extra_data
)
{
    if ((Transforms & SlotBit) != 0u) {
        if ((transform_mask & SlotBit) != 0u) {
            const uint location = transform_record_offset(feature_mask, transform_mask, SlotBit);
            const GLTFTextureUVTransform transform = extra_data.load<GLTFTextureUVTransform>(location);
            const float3 p = float3(uv, 1.0);
            return float2(dot(transform.row_u, p), dot(transform.row_v, p));
        }
    }
    return uv;
}

public float2 resolve_texcoord_sparse<let Transforms : uint, let NonzeroTexcoords : uint, let SlotBit : uint>(
    const SurfaceInteraction si,
    GLTFTextureInfo info,
    uint feature_mask,
    uint transform_mask,
    BufferPtr extra_data
)
{
    uint selector = 0u;
    if ((NonzeroTexcoords & SlotBit) != 0u)
        selector = info.texcoord();
    const float2 uv = surface_material::resolve_texcoord(si, selector);
    return transform_uv_sparse<Transforms, SlotBit>(uv, feature_mask, transform_mask, extra_data);
}

public float4 sample_texture_sparse<
    let Textures : uint,
    let Transforms : uint,
    let NonzeroTexcoords : uint,
    let SlotBit : uint,
    LodSampler : ILodSampler>(
    const SurfaceInteraction si,
    const LodSampler lod_sampler,
    GLTFTextureInfo info,
    uint feature_mask,
    uint transform_mask,
    BufferPtr extra_data,
    float4 fallback
)
{
    if ((Textures & SlotBit) != 0u) {
        if (info.handle.is_valid()) {
            const float2 uv = resolve_texcoord_sparse<Transforms, NonzeroTexcoords, SlotBit>(
                si,
                info,
                feature_mask,
                transform_mask,
                extra_data
            );
            return info.handle.sample(lod_sampler, uv, fallback);
        }
    }
    return fallback;
}

public Frame shading_frame_sparse<
    let Textures : uint,
    let Transforms : uint,
    let NonzeroTexcoords : uint,
    let SlotBit : uint,
    LodSampler : ILodSampler>(
    const SurfaceInteraction si,
    const LodSampler lod_sampler,
    const MaterialInstanceHints hints,
    MaterialFlags flags,
    GLTFTextureInfo info,
    float normal_scale,
    uint feature_mask,
    uint transform_mask,
    BufferPtr extra_data
)
{
    Frame frame = si.shading_frame_ws;
    if ((Textures & SlotBit) != 0u) {
        if (info.handle.is_valid() && surface_material::is_normal_mapping_enabled(hints)) {
            const float2 uv = resolve_texcoord_sparse<Transforms, NonzeroTexcoords, SlotBit>(
                si,
                info,
                feature_mask,
                transform_mask,
                extra_data
            );
            const float3 encoded = info.handle.sample(lod_sampler, uv, float4(0.5, 0.5, 1.0, 0.0)).xyz;
            float3 normal = surface_material::decode_normal_map(encoded, info.normal_map_encoded());
            normal.xy *= normal_scale;
            normal = dot(normal, normal) > 0.0 ? normalize(normal) : float3(0.0, 0.0, 1.0);
            frame = Frame(frame.to_global(normal), frame.tangent, frame.handedness_sign());
        }
    }
    const bool orient_backface = is_set(flags, MaterialFlags::two_sided);
    frame = surface_material::orient_shading_frame(si, frame, orient_backface);
    if ((hints & MaterialInstanceHints::adjust_shading_normal) != MaterialInstanceHints::none)
        frame = surface_material::adjust_shading_frame(si, frame, orient_backface);
    return frame;
}

// KHR_materials_clearcoat's simple layering model uses the view-normal
// Fresnel term, rather than the microfacet half-vector.
// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_clearcoat/README.md#layering
public float clearcoat_weight(float factor, float cosine)
{
    return saturate(factor) * (0.04 + 0.96 * pow(1.0 - saturate(abs(cosine)), 5.0));
}

public float2 transform_uv(float2 uv, uint binding, uint transform_mask, BufferPtr transform_data, uint byte_offset = 0)
{
    const uint bit = 1u << binding;
    if ((transform_mask & bit) == 0)
        return uv;
    const uint index = countbits(transform_mask & (bit - 1u));
    const GLTFTextureUVTransform transform
        = transform_data.load<GLTFTextureUVTransform>(byte_offset + index * sizeof(GLTFTextureUVTransform));
    const float3 p = float3(uv, 1.0);
    return float2(dot(transform.row_u, p), dot(transform.row_v, p));
}

float2 resolve_texcoord(
    const SurfaceInteraction si,
    uint texcoord,
    uint binding,
    uint transform_mask,
    BufferPtr transform_data,
    uint byte_offset = 0
)
{
    return transform_uv(
        surface_material::resolve_texcoord(si, texcoord),
        binding,
        transform_mask,
        transform_data,
        byte_offset
    );
}

Frame shading_frame<LodSampler : ILodSampler>(
    const SurfaceInteraction si,
    const LodSampler lod_sampler,
    const MaterialInstanceHints hints,
    MaterialFlags flags,
    TextureHandle normal_texture,
    float normal_scale,
    uint normal_texcoord,
    uint transform_mask,
    BufferPtr transform_data,
    uint byte_offset = 0,
    uint normal_binding = 2,
    bool normal_map_encoded = false
)
{
    Frame frame = si.shading_frame_ws;
    if (normal_texture.is_valid() && surface_material::is_normal_mapping_enabled(hints)) {
        const float2 uv
            = resolve_texcoord(si, normal_texcoord, normal_binding, transform_mask, transform_data, byte_offset);
        float3 encoded = normal_texture.sample(lod_sampler, uv, float4(0.5, 0.5, 1.0, 0.0)).xyz;
        float3 normal = surface_material::decode_normal_map(encoded, normal_map_encoded);
        // glTF scales the tangent-plane components; the normal component remains unchanged.
        // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#material-normaltexture
        normal.xy *= normal_scale;
        normal = dot(normal, normal) > 0.0 ? normalize(normal) : float3(0.0, 0.0, 1.0);
        frame = Frame(frame.to_global(normal), frame.tangent, frame.handedness_sign());
    }
    const bool orient_backface = is_set(flags, MaterialFlags::two_sided);
    frame = surface_material::orient_shading_frame(si, frame, orient_backface);
    if ((hints & MaterialInstanceHints::adjust_shading_normal) != MaterialInstanceHints::none)
        frame = surface_material::adjust_shading_frame(si, frame, orient_backface);
    return frame;
}

float opacity<LodSampler : ILodSampler>(
    const SurfaceInteraction si,
    const LodSampler lod_sampler,
    MaterialHeader header,
    TextureHandle texture,
    uint texcoord,
    bool use_vertex_color,
    uint transform_mask,
    BufferPtr transform_data,
    uint byte_offset = 0
)
{
    if (!is_set(header.opacity_flags, OpacityFlags::enabled))
        return 1.0;
    const float2 uv = resolve_texcoord(si, texcoord, 0, transform_mask, transform_data, byte_offset);
    float result = texture.sample(lod_sampler, uv, float4(1.0)).a * header.opacity_factor;
    if (use_vertex_color)
        result *= surface_material::resolve_vertex_color(si).a;
    return result;
}

float3 emission<LodSampler : ILodSampler>(
    const SurfaceInteraction si,
    const LodSampler lod_sampler,
    TextureHandle texture,
    float3 factor,
    uint texcoord,
    uint transform_mask,
    BufferPtr transform_data,
    uint byte_offset = 0
)
{
    const float2 uv = resolve_texcoord(si, texcoord, 3, transform_mask, transform_data, byte_offset);
    return texture.sample(lod_sampler, uv, float4(1.0)).rgb * factor;
}
}
