// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FALCOR_SHARED_HOST
implementing render;
import falcor2.utils;
#endif

namespace gltfmat {

// These masks are shared with the native packer. A feature bit says that work
// can be active for a material; only EXTENSION_BITS consume 32-byte tail slots.
namespace FeatureBits {
public static const uint NORMAL = 1u << 0;
public static const uint EMISSION = 1u << 1;
public static const uint VERTEX_COLOR = 1u << 2;
public static const uint MULTISCATTER = 1u << 3;
public static const uint SPECULAR = 1u << 4;
public static const uint TRANSMISSION = 1u << 5;
public static const uint IOR = 1u << 6;
public static const uint VOLUME = 1u << 7;
public static const uint CLEARCOAT = 1u << 8;
public static const uint SHEEN = 1u << 9;
public static const uint IRIDESCENCE = 1u << 10;
public static const uint ANISOTROPY = 1u << 11;
public static const uint DIFFUSE_TRANSMISSION = 1u << 12;

public static const uint EXTENSION_BITS
    = SPECULAR | TRANSMISSION | IOR | VOLUME | CLEARCOAT | SHEEN | IRIDESCENCE | ANISOTROPY | DIFFUSE_TRANSMISSION;
public static const uint ALL_MR = NORMAL | EMISSION | VERTEX_COLOR | MULTISCATTER | EXTENSION_BITS;
public static const uint ALL_SPEC_GLOSS = NORMAL | EMISSION | VERTEX_COLOR;
}

// Slot numbers also determine the order of sparse 24-byte UV transforms.
namespace TextureBits {
public static const uint BASE_COLOR = 1u << 0;
public static const uint METALLIC_ROUGHNESS = 1u << 1;
public static const uint NORMAL = 1u << 2;
public static const uint EMISSIVE = 1u << 3;
public static const uint SPECULAR = 1u << 4;
public static const uint SPECULAR_COLOR = 1u << 5;
public static const uint TRANSMISSION = 1u << 6;
public static const uint CLEARCOAT = 1u << 7;
public static const uint CLEARCOAT_ROUGHNESS = 1u << 8;
public static const uint CLEARCOAT_NORMAL = 1u << 9;
public static const uint SHEEN_COLOR = 1u << 10;
public static const uint SHEEN_ROUGHNESS = 1u << 11;
public static const uint IRIDESCENCE = 1u << 12;
public static const uint IRIDESCENCE_THICKNESS = 1u << 13;
public static const uint ANISOTROPY = 1u << 14;
public static const uint DIFFUSE_TRANSMISSION = 1u << 15;
public static const uint DIFFUSE_TRANSMISSION_COLOR = 1u << 16;
public static const uint ALL = (1u << 17) - 1u;
}

namespace SpecGlossTextureBits {
public static const uint DIFFUSE = 1u << 0;
public static const uint SPECULAR_GLOSSINESS = 1u << 1;
public static const uint NORMAL = 1u << 2;
public static const uint EMISSIVE = 1u << 3;
public static const uint ALL = DIFFUSE | SPECULAR_GLOSSINESS | NORMAL | EMISSIVE;
}

namespace TextureInfoBits {
public static const uint TEXCOORD_MASK = 7u;
public static const uint NORMAL_MAP_ENCODED = 1u << 3;
}

// Shared with the native material writers. UV selectors occupy three bits;
// the named flags retain their positions in the existing packed material data.
public struct TCBits
{
    public typedef BitField<uint, 0, 3> base_color;
    public typedef BitField<uint, 3, 3> metallic_roughness;
    public typedef BitField<uint, 6, 3> normal;
    public typedef BitField<uint, 9, 3> emissive;
    public typedef BitField<uint, 12, 3> specular;
    public typedef BitField<uint, 15, 3> specular_color;
    public typedef BitField<uint, 18, 3> transmission;
    public typedef BitField<uint, 21, 1> normal_map_encoded;
    public typedef BitField<uint, 30, 1> has_surface_extensions;
    public typedef BitField<uint, 31, 1> use_vertex_color;
};

public struct SpecGlossTCBits
{
    public typedef BitField<uint, 0, 3> diffuse;
    public typedef BitField<uint, 3, 3> specular_glossiness;
    public typedef BitField<uint, 6, 3> normal;
    public typedef BitField<uint, 9, 3> emissive;
    public typedef BitField<uint, 15, 1> use_vertex_color;
    public typedef BitField<uint, 16, 1> normal_map_encoded;
};

public struct ClearcoatTCBits
{
    public typedef BitField<uint, 0, 3> texture;
    public typedef BitField<uint, 3, 3> roughness;
    public typedef BitField<uint, 6, 3> normal;
    public typedef BitField<uint, 9, 1> normal_map_encoded;
};

public struct SheenTCBits
{
    public typedef BitField<uint, 0, 3> color;
    public typedef BitField<uint, 3, 3> roughness;
};

public struct IridescenceTCBits
{
    public typedef BitField<uint, 0, 3> texture;
    public typedef BitField<uint, 3, 3> thickness;
};

public struct SurfaceExtensionsTCBits
{
    public typedef BitField<uint, 0, 3> anisotropy;
    public typedef BitField<uint, 3, 3> diffuse_transmission;
    public typedef BitField<uint, 6, 3> diffuse_transmission_color;
};

} // namespace gltfmat
