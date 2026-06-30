// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Global scene data.
public ParameterBlock<Scene> g_scene;

public struct Scene
{
    /// Top-level acceleration structure.
    public RaytracingAccelerationStructure tlas;

    /// Triangle geometry data.
    public TriangleGeometryData triangle_geometry;

    /// Linear-swept-sphere geometry data.
    public LSSGeometryData lss_geometry;

    public StructuredBuffer<GeometryInstanceDesc> geometry_instance_descs;

    public MaterialSystem material_system;
    public LightSystem light_system;
    public EmissiveGeometry emissive_geometry;

    public StructuredBuffer<float4x4> world_from_object;
    public StructuredBuffer<float4x4> object_from_world;
    public StructuredBuffer<float4x4> prev_world_from_object;

    public GeometryInstanceDesc get_geometry_instance_desc(GeometryInstanceID id)
    {
        return geometry_instance_descs[uint(id)];
    }

    public GeometryType get_geometry_instance_type(GeometryInstanceID id)
    {
        return get_geometry_instance_desc(id).header.type;
    }

    public IMaterial get_material(MaterialID material_id) { return material_system.get_material(material_id); }

    public float4x4 get_world_from_object(TransformID id) { return world_from_object[uint(id)]; }

    public float4x4 get_world_from_object(GeometryInstanceID id)
    {
        return get_world_from_object(get_geometry_instance_desc(id).header.transform_id);
    }

    public float4x4 get_object_from_world(TransformID id) { return object_from_world[uint(id)]; }

    public float4x4 get_object_from_world(GeometryInstanceID id)
    {
        return object_from_world[uint(get_geometry_instance_desc(id).header.transform_id)];
    }

    public float4x4 get_prev_world_from_object(TransformID id) { return prev_world_from_object[uint(id)]; }

    public float4x4 get_prev_world_from_object(GeometryInstanceID id)
    {
        return get_prev_world_from_object(get_geometry_instance_desc(id).header.transform_id);
    }

    /// Compute the motion vector for a surface point, accounting for both camera and object motion.
    public float2
    get_motion_vector(Camera camera, Camera prev_camera, GeometryInstanceID geometry_instance_id, float3 position_ws)
    {
        float4x4 obj_from_world = get_object_from_world(geometry_instance_id);
        float3 position_os = mul(obj_from_world, float4(position_ws, 1.0)).xyz;
        float4x4 prev_wfo = get_prev_world_from_object(geometry_instance_id);
        float3 prev_position_ws = mul(prev_wfo, float4(position_os, 1.0)).xyz;
        float2 current_pixel = camera.project_world_to_pixel(position_ws);
        float2 prev_pixel = prev_camera.project_world_to_pixel(prev_position_ws);
        return prev_pixel - current_pixel;
    }

    /// Construct a surface interaction from a hit, dispatching by geometry type.
    public SurfaceInteraction make_surface_interaction(HitInfo hit, float3 wi_ws)
    {
        SurfaceInteraction sd = {};
        if (hit.geometry_type == GeometryType::triangle) {
            sd = make_triangle_surface_interaction(hit, wi_ws);
        } else if (hit.geometry_type == GeometryType::lss) {
            sd = make_lss_surface_interaction(hit, wi_ws);
        }
        return sd;
    }

    [Differentiable]
    public IMaterialInstance get_material_instance<LodSampler : ILodSampler>(
        const HitInfo hit,
        const SurfaceInteraction si,
        LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialID material_id = get_geometry_instance_desc(hit.geometry_instance_id).header.material_id;
        let material = get_material(material_id);
        return material.setup_material_instance(si, lod_sampler, hints);
    }

    public IMaterial get_material(const HitInfo hit)
    {
        MaterialID material_id = get_geometry_instance_desc(hit.geometry_instance_id).header.material_id;
        return material_system.get_material(material_id);
    }

    public MaterialHeader get_material_header(const HitInfo hit)
    {
        MaterialID material_id = get_geometry_instance_desc(hit.geometry_instance_id).header.material_id;
        return material_system.get_material_data(material_id).header;
    }

    // Lights

    /// Returns true if the scene has any analytic lights.
    public bool has_analytic_lights() { return light_system.has_any_analytic_lights(); }

    /// Returns true if the scene has an environment light.
    public bool has_environment_light() { return light_system.has_environment_light(); }

    /// Returns true if the scene has emissive triangles.
    public bool has_emissive_triangles() { return emissive_geometry.has_active_triangles(); }
};
