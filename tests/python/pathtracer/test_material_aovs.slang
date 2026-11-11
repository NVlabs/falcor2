// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

static float4 written_value;
static uint selected_id;
static bool output_enabled;

public struct ProbeWriter : IMaterialAOVWriter
{
    public static bool generic_data_enabled(uint id) { return output_enabled && id == selected_id; }
    public static void generic_data(uint2 coord, uint id, float4 value) { written_value = value; }
}

public float4 probe(uint geometry_id, uint output_id, bool enabled)
{
    TriangleHitInfo hit = TriangleHitInfo(GeometryInstanceID(geometry_id), 0, float2(.2, .3));
    const SurfaceInteraction si = g_scene.make_surface_interaction(hit, float3(0, 0, -1));
    written_value = float4(-1);
    selected_id = output_id;
    output_enabled = enabled;
    g_scene.get_material(hit).write_material_aovs<ProbeWriter>(
        uint2(0),
        si,
        ExplicitLodSampler(0),
        MaterialInstanceHints::adjust_shading_normal
    );
    return written_value;
}
