// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

public struct StandardNormalRepairResult
{
    float3 original_normal;
    float3 adjusted_normal;
    float3 tangent;
    float3 bitangent;
    float3 eval;
};

public StandardNormalRepairResult
standard_normal_repair_probe(uint material_id, float2 uv, float side, float handedness)
{
    SurfaceInteraction si = {};
    si.normal_ws = float3(0.0, 0.0, 1.0);
    si.wi_ws = side * si.normal_ws;
    si.front_facing = side > 0.0;
    si.uv = uv;
    si.shading_frame_ws = Frame(float3(sqrt(0.75), 0.0, 0.5), float3(0.5, 0.0, -sqrt(0.75)), handedness);
    si.exterior_ior = 1.0;
    let material = g_scene.get_material(MaterialID(material_id));
    let original = material.setup_material_instance(si, ExplicitLodSampler(0.0), MaterialInstanceHints::none);
    let adjusted
        = material.setup_material_instance(si, ExplicitLodSampler(0.0), MaterialInstanceHints::adjust_shading_normal);
    StandardNormalRepairResult result;
    result.original_normal = original.collect_properties(si).guide_normal;
    result.adjusted_normal = adjusted.collect_properties(si).guide_normal;
    let properties = adjusted.collect_extra_bsdf_properties(si, si.wi_ws).value;
    result.tangent = si.shading_frame_ws.to_global(properties.bsdf_T[0]);
    result.bitangent = si.shading_frame_ws.to_global(properties.bsdf_B[0]);
    var sg = TinyUniformSampleGenerator(uint2(7, 0), 0);
    result.eval = adjusted.eval(si, si.wi_ws, sg);
    return result;
}
