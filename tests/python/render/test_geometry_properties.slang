// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

public PackedHitInfo packed_hit_info_probe(uint geometry_instance_id, float2 barycentrics)
{
    const TriangleHitInfo hit = TriangleHitInfo(GeometryInstanceID(geometry_instance_id), 0, barycentrics);
    const SurfaceInteraction si = g_scene.make_surface_interaction(hit, float3(0.0, 0.0, 1.0));
    if (let packed_hit_info = si.hit_info.get())
        return packed_hit_info;
    return {};
}

public struct AdditionalAttributeProbe
{
    public uint mask;
    public uint byte_offset;
    public uint format;
    public uint data[12];
};

public AdditionalAttributeProbe
additional_attribute_probe(uint geometry_instance_id, uint attribute_mask_bit, uint word_count)
{
    const GeometryInstanceDesc geometry_instance
        = g_scene.get_geometry_instance_desc(GeometryInstanceID(geometry_instance_id));
    const TriangleGeometryInstanceDesc triangle_geometry_instance
        = reinterpret<TriangleGeometryInstanceDesc>(geometry_instance);
    const uint geometry_id = uint(triangle_geometry_instance.header.geometry_id);
    const uint attribute_desc_offset = geometry_id * ADDITIONAL_ATTRIBUTE_DESC_ROW_SIZE;
    const uint attribute_mask = g_scene.triangle_geometry.additional_attributes[attribute_desc_offset].byte_offset;
    const uint lower_attribute_mask = attribute_mask & ((1u << attribute_mask_bit) - 1u);
    const uint attribute_rank = countbits(lower_attribute_mask);
    const PackedAdditionalAttribute attribute
        = g_scene.triangle_geometry.additional_attributes[attribute_desc_offset + 1 + attribute_rank];

    AdditionalAttributeProbe result = {};
    result.mask = attribute_mask;
    result.byte_offset = attribute.byte_offset;
    result.format = attribute.format;
    for (uint index = 0; index < word_count; ++index)
        result.data[index]
            = g_scene.triangle_geometry.additional_attribute_data.Load(attribute.byte_offset + index * 4);
    return result;
}

public struct AttributeAccessorProbe
{
    public uint success_mask;
    public uint int_vector_success_mask;
    public uint bool_vector_success_mask;
    public uint additional_texcoord_success_mask;
    public bool color_success;
    public int value_int;
    public int2 value_int2;
    public int3 value_int3;
    public int4 value_int4;
    public bool value_bool;
    public bool2 value_bool2;
    public bool3 value_bool3;
    public bool4 value_bool4;
    public float value_float;
    public float2 value2;
    public float3 value3;
    public float4 value4;
    public float2 additional_texcoord2;
    public float3 additional_texcoord3;
    public float4 additional_texcoord4;
    public float4 color;
    public float3 object_origin_ws;
    public float3 world_origin_os;
};

public struct AttributeFallbackProbe
{
    public float3 generic_zero;
    public float3 generic_default;
    public float2 texcoord2_default;
    public float3 texcoord3_default;
    public float4 texcoord4_default;
    public float4 color_default;
    public float3 identity_point;
};

public AttributeFallbackProbe attribute_fallback_probe(uint unused)
{
    const SurfaceInteraction si = {};
    const IGeomPropAccessor accessor = si.get_geomprop_accessor();

    AttributeFallbackProbe result;
    result.generic_zero = accessor.get_float3(99);
    result.generic_default = accessor.get_float3(99, float3(1.0, 2.0, 3.0));
    result.texcoord2_default = accessor.get_additional_texcoord(1, float2(4.0, 5.0));
    result.texcoord3_default = accessor.get_additional_texcoord(1, float3(6.0, 7.0, 8.0));
    result.texcoord4_default = accessor.get_additional_texcoord(1, float4(9.0, 10.0, 11.0, 12.0));
    result.color_default = accessor.get_color(0, float4(13.0, 14.0, 15.0, 16.0));
    result.identity_point = transform_point(
        accessor.get_space_transform(GeomPropSpace::object, GeomPropSpace::world),
        float3(17.0, 18.0, 19.0)
    );
    return result;
}

AttributeAccessorProbe read_attribute_accessor(const SurfaceInteraction si, uint attribute_id)
{
    IGeomPropAccessor accessor = si.get_geomprop_accessor();

    AttributeAccessorProbe result = {};
    if (accessor.try_get(attribute_id, result.value_int))
        result.success_mask |= 1u << 0;
    if (accessor.try_get(attribute_id, result.value_int2))
        result.int_vector_success_mask |= 1u << 0;
    if (accessor.try_get(attribute_id, result.value_int3))
        result.int_vector_success_mask |= 1u << 1;
    if (accessor.try_get(attribute_id, result.value_int4))
        result.int_vector_success_mask |= 1u << 2;
    if (accessor.try_get(attribute_id, result.value_bool))
        result.success_mask |= 1u << 1;
    if (accessor.try_get(attribute_id, result.value_bool2))
        result.bool_vector_success_mask |= 1u << 0;
    if (accessor.try_get(attribute_id, result.value_bool3))
        result.bool_vector_success_mask |= 1u << 1;
    if (accessor.try_get(attribute_id, result.value_bool4))
        result.bool_vector_success_mask |= 1u << 2;
    if (accessor.try_get(attribute_id, result.value_float))
        result.success_mask |= 1u << 2;
    if (accessor.try_get(attribute_id, result.value2))
        result.success_mask |= 1u << 3;
    if (accessor.try_get(attribute_id, result.value3))
        result.success_mask |= 1u << 4;
    if (accessor.try_get(attribute_id, result.value4))
        result.success_mask |= 1u << 5;
    if (accessor.try_get_additional_texcoord(attribute_id, result.additional_texcoord2))
        result.additional_texcoord_success_mask |= 1u << 0;
    if (accessor.try_get_additional_texcoord(attribute_id, result.additional_texcoord3))
        result.additional_texcoord_success_mask |= 1u << 1;
    if (accessor.try_get_additional_texcoord(attribute_id, result.additional_texcoord4))
        result.additional_texcoord_success_mask |= 1u << 2;
    result.color_success = accessor.try_get_color(0, result.color);

    result.object_origin_ws
        = transform_point(accessor.get_space_transform(GeomPropSpace::world, GeomPropSpace::object), float3(0.f));
    result.world_origin_os
        = transform_point(accessor.get_space_transform(GeomPropSpace::object, GeomPropSpace::world), float3(0.f));
    return result;
}

public AttributeAccessorProbe
attribute_accessor_probe(uint geometry_instance_id, uint attribute_id, float2 barycentrics)
{
    const TriangleHitInfo hit = TriangleHitInfo(GeometryInstanceID(geometry_instance_id), 0, barycentrics);
    const SurfaceInteraction si = g_scene.make_surface_interaction(hit, float3(0.f, 0.f, 1.f));
    return read_attribute_accessor(si, attribute_id);
}

public float2
imported_texcoord_probe(uint geometry_instance_id, uint primitive_index, float2 barycentrics, uint texture_coord_index)
{
    const TriangleHitInfo hit
        = TriangleHitInfo(GeometryInstanceID(geometry_instance_id), primitive_index, barycentrics);
    const SurfaceInteraction si = g_scene.make_surface_interaction(hit, float3(0.f, 0.f, 1.f));
    if (texture_coord_index == 0)
        return si.uv;

    float2 texture_coord = {};
    si.get_geomprop_accessor().try_get_additional_texcoord(texture_coord_index, texture_coord);
    return texture_coord;
}

public AttributeAccessorProbe imported_attribute_accessor_probe(
    uint geometry_instance_id,
    uint primitive_index,
    uint attribute_id,
    float2 barycentrics
)
{
    const TriangleHitInfo hit
        = TriangleHitInfo(GeometryInstanceID(geometry_instance_id), primitive_index, barycentrics);
    const SurfaceInteraction si = g_scene.make_surface_interaction(hit, float3(0.f, 0.f, 1.f));
    return read_attribute_accessor(si, attribute_id);
}

public AttributeAccessorProbe
lss_attribute_accessor_probe(uint geometry_instance_id, uint primitive_index, uint attribute_id, float2 uv)
{
    PackedHitInfo packed_hit = {};
    packed_hit.geometry_type = GeometryType::lss;
    packed_hit.geometry_instance_id = GeometryInstanceID(geometry_instance_id);
    packed_hit.payload[0] = primitive_index;
    packed_hit.payload[1] = asuint(uv.x);
    packed_hit.payload[2] = asuint(uv.y);
    const LSSHitInfo hit = LSSHitInfo(packed_hit);
    const SurfaceInteraction si = g_scene.make_lss_surface_interaction(hit, float3(0.f, 0.f, 1.f));
    return read_attribute_accessor(si, attribute_id);
}

public struct MultiUVMaterialProbe
{
    public float3 emissive_radiance;
    public float3 material_color;
    public float3 diffuse_reflection_albedo;
    public float3 guide_normal;
    public float opacity;
    public float interior_ior;
};

MultiUVMaterialProbe read_material_hit(SurfaceInteraction si, IMaterial material)
{
    MultiUVMaterialProbe result = {};
    const ExplicitLodSampler lod_sampler = ExplicitLodSampler(0);
    let material_instance = material.setup_material_instance(si, lod_sampler);
    const MaterialProperties properties = material_instance.collect_properties(si);
    result.emissive_radiance = material.eval_emissive_radiance(si, lod_sampler);
    result.material_color = properties.material_color;
    result.diffuse_reflection_albedo = properties.diffuse_reflection_albedo;
    result.guide_normal = properties.guide_normal;
    result.opacity = material.eval_opacity(si, lod_sampler);
    result.interior_ior = material_instance.get_interior_ior();
    return result;
}

public MultiUVMaterialProbe
multi_uv_material_probe(uint geometry_instance_id, uint primitive_index, float2 barycentrics)
{
    const TriangleHitInfo hit
        = TriangleHitInfo(GeometryInstanceID(geometry_instance_id), primitive_index, barycentrics);
    return read_material_hit(g_scene.make_surface_interaction(hit, float3(0, 0, 1)), g_scene.get_material(hit));
}

public MultiUVMaterialProbe lss_material_probe(uint geometry_instance_id, uint primitive_index, float2 uv)
{
    PackedHitInfo packed_hit = {};
    packed_hit.geometry_type = GeometryType::lss;
    packed_hit.geometry_instance_id = GeometryInstanceID(geometry_instance_id);
    packed_hit.payload[0] = primitive_index;
    packed_hit.payload[1] = asuint(uv.x);
    packed_hit.payload[2] = asuint(uv.y);
    const LSSHitInfo hit = LSSHitInfo(packed_hit);
    return read_material_hit(g_scene.make_lss_surface_interaction(hit, float3(0, 0, 1)), g_scene.get_material(hit));
}

public MultiUVMaterialProbe synthetic_material_probe(uint geometry_instance_id, uint primitive_index, float2 uv)
{
    const TriangleHitInfo hit = TriangleHitInfo(GeometryInstanceID(geometry_instance_id), primitive_index, uv);
    SurfaceInteraction si;
    si.uv = uv;
    si.wi_ws = float3(0, 0, 1);
    si.normal_ws = float3(0, 0, 1);
    si.shading_frame_ws = Frame(si.normal_ws);
    si.front_facing = true;
    return read_material_hit(si, g_scene.get_material(hit));
}

public float multi_uv_opacity_probe(uint geometry_instance_id, uint primitive_index, float2 barycentrics)
{
    const TriangleHitInfo hit
        = TriangleHitInfo(GeometryInstanceID(geometry_instance_id), primitive_index, barycentrics);
    const SurfaceInteraction si = g_scene.make_surface_interaction(hit, float3(0.f, 0.f, 1.f));
    return g_scene.get_material(hit).eval_opacity(si, ExplicitLodSampler(0));
}
