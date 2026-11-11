// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

StructuredBuffer<GeometryInstanceDesc> instance_descs;
StructuredBuffer<float4x4> current_inverse;
StructuredBuffer<float4x4> previous_world;
StructuredBuffer<BufferPtr> geometry;

public float3 previous_position(bool curve, uint primitive, float2 parameters, float3 position_ws)
{
    Scene scene = g_scene;
    scene.geometry_instance_descs = instance_descs;
    scene.object_from_world = current_inverse;
    scene.prev_world_from_object = previous_world;
    scene.geometry = geometry;
    if (curve) {
        LSSHitInfo hit;
        hit.geometry_instance_id = GeometryInstanceID(0);
        hit.primitive_index = primitive;
        hit.curve_parameters = parameters;
        return scene.get_previous_position_ws(hit, position_ws);
    }
    return scene.get_previous_position_ws(TriangleHitInfo(GeometryInstanceID(0), primitive, parameters), position_ws);
}

public float2 motion(float3 current_position, float3 previous_position, float3 previous_camera_position)
{
    Camera camera = {};
    camera.dims = uint2(100, 80);
    camera.image_u = float3(1.0, 0.0, 0.0);
    camera.image_v = float3(0.0, 1.0, 0.0);
    camera.image_w = float3(0.0, 0.0, 1.0);
    Camera previous_camera = camera;
    previous_camera.position = previous_camera_position;
    return camera.get_motion_vector(previous_camera, current_position, previous_position);
}

public uint2 static_descriptor_flags(uint instance_id)
{
    let instance = g_scene.get_geometry_instance_desc(GeometryInstanceID(instance_id));
    uint geometry_count;
    if (instance.header.type == GeometryType::triangle) {
        geometry_count = g_scene.triangle_geometry.descs[uint(instance.header.geometry_id)].index_count;
    } else {
        geometry_count = g_scene.lss_geometry.descs[uint(instance.header.geometry_id)].index_count;
    }
    return uint2(geometry_count, uint(instance.header.flags));
}

public float3 previous_area_position(float3 position_ws)
{
    Scene scene = g_scene;
    scene.geometry_instance_descs = instance_descs;
    scene.object_from_world = current_inverse;
    scene.prev_world_from_object = previous_world;
    return scene.get_previous_position_ws(AreaLightHitInfo(GeometryInstanceID(0)), position_ws);
}
