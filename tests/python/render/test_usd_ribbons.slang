// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;
import falcor2.ui.kernels.scene_picker;

public struct RibbonProbe
{
    public bool valid;
    public float3 position;
    public float3 normal;
    public float3 shading_normal;
    public float2 uv;
    public float3 tangent;
    public float3 emission;
};

public RibbonProbe ribbon_probe(float3 origin, float3 direction)
{
    RayTracingSceneIntersector intersector;
    const IHitInfo hit = intersector.intersect(Ray(origin, direction));
    RibbonProbe result = {};
    result.valid = hit.valid;
    if (hit.valid) {
        const SurfaceInteraction si = g_scene.make_surface_interaction(hit, -direction);
        result.position = si.position_ws;
        result.normal = si.normal_ws;
        result.shading_normal = si.shading_frame_ws.normal;
        result.uv = si.uv;
        result.tangent = si.shading_frame_ws.tangent;
        result.emission = g_scene.get_material(hit).eval_emissive_radiance(si, ExplicitLodSampler(0));
    }
    return result;
}
