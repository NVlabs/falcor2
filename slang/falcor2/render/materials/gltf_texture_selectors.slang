// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FALCOR_SHARED_HOST
implementing render;
import falcor2.utils;
#endif

namespace gltfmat {

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
