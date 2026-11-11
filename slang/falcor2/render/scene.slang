// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Global scene data.
public ParameterBlock<Scene> g_scene;

public struct Scene
{
    /// Top-level acceleration structure.
    public RaytracingAccelerationStructure tlas;

    /// Shared table of homogeneous vertex and index buffers.
    public StructuredBuffer<BufferPtr> geometry;

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

    /// Get the geometry-instance ID carried by a real surface interaction.
    /// Returns false and GeometryInstanceID::invalid for synthetic interactions.
    public bool try_get_geometry_instance_id(const SurfaceInteraction si, out GeometryInstanceID id)
    {
        id = GeometryInstanceID::invalid;
        if (let packed_hit = si.hit_info.get()) {
            if (packed_hit.geometry_type != GeometryType::invalid) {
                id = packed_hit.geometry_instance_id;
                return true;
            }
        }
        return false;
    }

    /// Get the world-from-object transform for the geometry instance at a surface interaction.
    public float4x4
    get_geometry_world_from_object(const SurfaceInteraction si, const float4x4 fallback = GEOMETRY_IDENTITY_4X4)
    {
        GeometryInstanceID id;
        return try_get_geometry_instance_id(si, id) ? get_world_from_object(id) : fallback;
    }

    /// Get the object-from-world transform for the geometry instance at a surface interaction.
    public float4x4
    get_geometry_object_from_world(const SurfaceInteraction si, const float4x4 fallback = GEOMETRY_IDENTITY_4X4)
    {
        GeometryInstanceID id;
        return try_get_geometry_instance_id(si, id) ? get_object_from_world(id) : fallback;
    }

    public float4x4 get_prev_world_from_object(TransformID id) { return prev_world_from_object[uint(id)]; }

    public float4x4 get_prev_world_from_object(GeometryInstanceID id)
    {
        return get_prev_world_from_object(get_geometry_instance_desc(id).header.transform_id);
    }

    /// Resolve the same surface point in previous-frame world space.
    /// Static vertices reuse position_ws; vertex motion reconstructs the previous object-space point from hit.
    public float3 get_previous_position_ws(IHitInfo hit, float3 position_ws)
    {
        let instance = get_geometry_instance_desc(hit.geometry_instance_id);
        float3 prev_position_os;
        if (!is_set(instance.header.flags, GeometryInstanceFlags::vertex_motion)) {
            prev_position_os = transform_point(get_object_from_world(instance.header.transform_id), position_ws);
        } else if (let triangle_hit = hit as TriangleHitInfo) {
            prev_position_os = get_previous_position_os(triangle_hit);
        } else if (let lss_hit = hit as LSSHitInfo) {
            prev_position_os = get_previous_position_os(lss_hit);
        } else {
            prev_position_os = transform_point(get_object_from_world(instance.header.transform_id), position_ws);
        }
        return transform_point(get_prev_world_from_object(instance.header.transform_id), prev_position_os);
    }

    /// Construct a surface interaction from a hit.
    /// Enable intrinsics only for the current native LSS hit in a ray-tracing hit shader.
    /// Cached hits, ray queries and procedural LSS hits must keep the default buffer-backed reconstruction.
    public SurfaceInteraction
    make_surface_interaction<let HasTracingIntrinsics : bool = false>(IHitInfo hit, float3 wi_ws)
    {
        if (let triangle_hit = hit as TriangleHitInfo) {
            return make_triangle_surface_interaction(triangle_hit, wi_ws);
        } else if (let lss_hit = hit as LSSHitInfo) {
            return make_lss_surface_interaction<HasTracingIntrinsics>(lss_hit, wi_ws);
        }
        return {};
    }

    [Differentiable]
    public IMaterialInstance get_material_instance<LodSampler : ILodSampler>(
        const IHitInfo hit,
        const SurfaceInteraction si,
        LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialID material_id = get_geometry_instance_desc(hit.geometry_instance_id).header.material_id;
        let material = get_material(material_id);
        return material.setup_material_instance(si, lod_sampler, hints);
    }

    public IMaterial get_material(const IHitInfo hit)
    {
        MaterialID material_id = get_geometry_instance_desc(hit.geometry_instance_id).header.material_id;
        return material_system.get_material(material_id);
    }

    public MaterialHeader get_material_header(const IHitInfo hit)
    {
        MaterialID material_id = get_geometry_instance_desc(hit.geometry_instance_id).header.material_id;
        return material_system.get_material_data(material_id).header;
    }

    // Lights

    /// Returns true if the scene has any analytic lights.
    public bool has_analytic_lights() { return light_system.has_any_analytic_lights(); }

    /// Returns true if the scene has one or more environment lights.
    public bool has_environment_lights() { return light_system.has_environment_lights(); }

    /// Returns true if the scene has emissive triangles.
    public bool has_emissive_triangles() { return emissive_geometry.has_active_triangles(); }
};
