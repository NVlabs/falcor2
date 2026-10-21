// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

public uint candidate_triangle_count()
{
    return g_scene.emissive_geometry.triangle_count;
}

public float4x4 probe(uint geometry_id, float3 outgoing)
{
    TriangleHitInfo hit = TriangleHitInfo(GeometryInstanceID(geometry_id), 0, float2(.2, .3));
    SurfaceInteraction si = g_scene.make_surface_interaction(hit, outgoing);
    let material = g_scene.get_material(hit);
    let instance = material.setup_material_instance(si, ExplicitLodSampler(0));
    float2 uv[3] = { float2(0), float2(1, 0), float2(0, 1) };
    return float4x4(
        float4(instance.eval_emissive_radiance(), float(instance.is_emissive())),
        float4(material.eval_emissive_radiance(si, ExplicitLodSampler(0)), float(material.is_emissive())),
        float4(material.eval_emission(si, ExplicitLodSampler(0)), material.compute_emission_sampling_grid(uv).x),
        float4(instance.collect_properties(si).emission, g_scene.emissive_geometry.active_triangle_count)
    );
}
