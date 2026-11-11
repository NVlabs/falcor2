// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct AreaLightHitInfo : IHitInfo
{
    public property GeometryType geometry_type
    {
        get {
            return GeometryType::area_light;
        }
    }

    public GeometryInstanceID geometry_instance_id;
    public property uint primitive_index
    {
        get {
            return 0;
        }
    }
    public float ray_t;

    public property bool valid
    {
        get {
            return true;
        }
    }

    public __init(GeometryInstanceID geometry_instance_id, float ray_t = 0.0)
    {
        this.geometry_instance_id = geometry_instance_id;
        this.ray_t = ray_t;
    }

    public __init(const PackedHitInfo p)
    {
        geometry_instance_id = p.geometry_instance_id;
        ray_t = asfloat(p.payload[1]);
    }

    public PackedHitInfo pack()
    {
        PackedHitInfo p = {};
        p.geometry_type = geometry_type;
        p.geometry_instance_id = geometry_instance_id;
        p.payload[1] = asuint(ray_t);
        return p;
    }
};

public struct AreaLightIntersectionAttributes
{
    float dummy = {};
};

public LightID get_area_light_id(const GeometryInstanceID geometry_instance_id)
{
    const AreaLightGeometryInstanceDesc desc
        = reinterpret<AreaLightGeometryInstanceDesc>(g_scene.get_geometry_instance_desc(geometry_instance_id));
    return desc.light_id;
}

public bool is_area_light_visible(const GeometryInstanceID geometry_instance_id)
{
    const AreaLightGeometryInstanceDesc desc
        = reinterpret<AreaLightGeometryInstanceDesc>(g_scene.get_geometry_instance_desc(geometry_instance_id));
    return is_set(desc.flags, AreaLightGeometryFlags::visible);
}

public struct DeferredAreaLightHitInfo : IDeferredHitInfo
{
    public AreaLightHitInfo create_hit()
    {
        return AreaLightHitInfo(GeometryInstanceID(InstanceID() + GeometryIndex2()), RayTCurrent());
    }

    public IHitInfo create() { return create_hit(); }
};

public bool intersect_area_light(
    const GeometryInstanceID geometry_instance_id,
    const float3 ray_origin_ws,
    const float3 ray_dir_ws,
    const float t_min,
    const float t_max,
    out float t
)
{
    t = {};
    const LightID light_id = get_area_light_id(geometry_instance_id);
    if (light_id == LightID::invalid)
        return false;
    ILight light = g_scene.light_system.get_light(light_id);
    if (!light.intersect_surface(Ray(ray_origin_ws, ray_dir_ws, t_min, t_max), t))
        return false;

    // Hidden light surfaces are additive emitter events. Skip their non-emitting
    // back faces instead of returning a no-op continuation hit.
    if (!is_area_light_visible(geometry_instance_id)) {
        const float3 pos_ws = ray_origin_ws + t * ray_dir_ws;
        const float3 normal_ws = light.eval_surface_normal(pos_ws);
        if (dot(normal_ws, ray_origin_ws - pos_ws) <= 0.0)
            return false;
    }
    return true;
}
