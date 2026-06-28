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
    /// Index of refraction for the medium on the front-facing side (i.e. "outside" the material at the hit).
    public float ior = 1.0;

    public __init() { }

    /// Returns the geometric normal in world-space such that it points towards the viewer.
    public float3 front_normal_ws() { return front_facing ? normal_ws : -normal_ws; }

    [mutating]
    public void adjust_shading_normal() { adjust_shading_normal(shading_frame_ws, normal_ws, wi_ws); }
};

interface IVolumeInteraction : IInteraction
{
};
