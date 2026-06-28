// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.mx139;
import falcor2.utils;

#include "../../../src/falcor2/render/material/materialx/codegen/materialx_1_39/libraries/stdlib/genslangpt/mx_normalmap.slang"

SurfaceInteraction make_signed_si(float handedness)
{
    SurfaceInteraction si = {};
    si.wi_ws = float3(0.0, 0.0, 1.0);
    si.normal_ws = float3(0.0, 0.0, 1.0);
    si.front_facing = true;
    si.uv = float2(0.5);
    si.shading_frame_ws = Frame(float3(0.0, 0.0, 1.0), float3(1.0, 0.0, 0.0), handedness);
    return si;
}

float material_bitangent_y(uint32_t material_id, SurfaceInteraction si)
{
    ExplicitLodSampler lod_sampler = ExplicitLodSampler(0.0);
    let material = g_scene.get_material(MaterialID(material_id));
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    // The material frame is observed through the common extra-BSDF reporting path so that Standard,
    // SpecGloss, and OpenPBR all use the same scene-material dispatch and material-instance setup.
    let maybe_properties = material_instance.collect_extra_bsdf_properties(si, float3(0.0, 0.0, 1.0));
    if (!maybe_properties.hasValue || maybe_properties.value.bsdf_count == 0)
        return -42.0;

    return maybe_properties.value.bsdf_B[0].y;
}

float4 material_frame_probe(
    uint32_t standard_material_id,
    uint32_t spec_gloss_material_id,
    uint32_t openpbr_material_id,
    float handedness
)
{
    SurfaceInteraction si = make_signed_si(handedness);

    return float4(
        material_bitangent_y(standard_material_id, si),
        material_bitangent_y(spec_gloss_material_id, si),
        material_bitangent_y(openpbr_material_id, si),
        mx139::mx_geomprop_bitangent(si).y
    );
}

float2 materialx_normalmap_probe(float handedness)
{
    SurfaceInteraction si = make_signed_si(handedness);
    float3 normalmap_result;

    // This normal-map sample points halfway between local +Y and local +Z. Its world Y component
    // therefore changes sign only when the supplied tangent-space bitangent handedness is used.
    mx_normalmap_vector2(
        float3(0.5, 0.5 + 0.5 * sqrt(0.5), 0.5 + 0.5 * sqrt(0.5)),
        float2(1.0),
        si.shading_frame_ws.normal,
        si.shading_frame_ws.tangent,
        si.shading_frame_ws.bitangent,
        normalmap_result
    );

    return float2(mx139::mx_geomprop_bitangent(si).y, normalmap_result.y);
}
