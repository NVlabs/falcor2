// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

namespace stdmat {
struct TCBits
{
    typedef BitField<uint, 0, 3> base_color;
    typedef BitField<uint, 0, 3> diffuse;
    typedef BitField<uint, 3, 3> metallic_roughness;
    typedef BitField<uint, 3, 3> specular_glossiness;
    typedef BitField<uint, 6, 3> normal;
    typedef BitField<uint, 9, 3> emissive;
    typedef BitField<uint, 12, 3> transmission;
    typedef BitField<uint, 15, 1> use_vertex_color;
    typedef BitField<uint, 16, 1> normal_map_encoded;
}

float get_metallic(const float3 diffuse, const float3 specular)
{
    float d = luminance(diffuse);
    float s = luminance(specular);
    if (s == 0.0)
        return 0.0;

    float b = s + d - 0.08;
    float c = 0.04 - s;
    float root = sqrt(max(b * b - 0.16 * c, 0.0));
    return saturate((root - b) * 12.5);
}

}
