// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public interface IInteraction : IDifferentiable
{
    /// Incident direction (i.e. -ray.dir) in world-space.
    public property float3 wi_ws;
    /// Position in world-space.
    public property float3 position_ws;
};

public interface ISurfaceInteraction : IInteraction
{
    /// Geometric normal in world-space.
    public property no_diff float3 normal_ws;
    /// True if surface is seen from the front (outside), false when seen from the back (inside).
    public property bool front_facing;
    /// Interpolated shading-frame in world-space.
    public property Frame shading_frame_ws;
};

/// Apply the renderer's spawn policy; an unavailable bound uses the standard offset.
public float3
offset_surface_ray_origin(float3 position, float3 oriented_normal, Conditional<float, ENABLE_PRECISE_RAY_OFFSETS> bound)
{
    if (let offset = bound.get()) {
        if (offset > 0.0)
            return offset_ray_origin_precise(position, oriented_normal, offset);
    }
    return offset_ray_origin(position, oriented_normal);
}

public struct SurfaceInteraction : ISurfaceInteraction
{
    /// Incident direction (i.e. -ray.dir) in world-space.
    public float3 wi_ws = {};
    /// Position in world-space.
    public float3 position_ws = {};
    /// Geometric normal in world-space.
    public no_diff float3 normal_ws = {};
    /// True if surface is seen from the front (outside), false when seen from the back (inside).
    public bool front_facing = {};
    /// Interpolated shading-frame in world-space.
    public Frame shading_frame_ws = {};

    /// Texture coordinates.
    public float2 uv = {};
    /// Index of refraction for the medium on the front-facing, exterior side of the material at the hit.
    public float exterior_ior = 1.0;

    /// Packed integer addressing data for scene-backed material inputs.
    public no_diff Conditional<PackedHitInfo, ENABLE_GEOMETRY_PROPERTIES> hit_info
        = Conditional<PackedHitInfo, ENABLE_GEOMETRY_PROPERTIES>(PackedHitInfo());

    /// World-space offset along the unit geometric normal; zero if unavailable.
    /// Compiled out when disabled. This is not stored in persistent path state.
    public no_diff Conditional<float, ENABLE_PRECISE_RAY_OFFSETS> ray_offset
        = Conditional<float, ENABLE_PRECISE_RAY_OFFSETS>(0.0);

    public __init() { }

    /// Store the packed addressing data when surface interactions include hit information.
    [mutating]
    public void set_hit_info(IHitInfo hit)
    {
        if (ENABLE_GEOMETRY_PROPERTIES)
            hit_info.set(hit.pack());
    }

    /// Returns the geometric normal in world-space such that it points towards the viewer.
    public float3 front_normal_ws() { return front_facing ? normal_ws : -normal_ws; }

    [mutating]
    public void adjust_shading_normal() { adjust_shading_normal(shading_frame_ws, normal_ws, wi_ws); }
};

interface IVolumeInteraction : IInteraction
{
};
