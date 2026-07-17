// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module scene_material_sampling_tools;

__exported import falcor2.render;
__exported import falcor2.utils;

public SurfaceInteraction setup_surface_interaction(float2 uv, float3 wi)
{
    SurfaceInteraction result;
    result.position_ws = float3(uv, 0);
    result.wi_ws = wi;
    result.uv = uv;

    result.shading_frame_ws = Frame(float3(0, 1, 0));
    result.normal_ws = result.shading_frame_ws.normal;
    result.front_facing = dot(result.wi_ws, result.normal_ws) >= 0;
    result.ior = 1.0;

    return result;
}

public float3
material_eval<SampleGenerator : ISampleGenerator>(uint3 seeds, IMaterial material, SurfaceInteraction si, float3 wo)
{
    var sg = SampleGenerator(seeds.xy, seeds.z);
    let lod_sampler = ExplicitLodSampler(0);
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);
    return material_instance.eval(si, wo, sg);
}

public MaterialSample
material_sample<SampleGenerator : ISampleGenerator>(uint3 seeds, IMaterial material, SurfaceInteraction si)
{
    var sg = SampleGenerator(seeds.xy, seeds.z);
    let lod_sampler = ExplicitLodSampler(0);
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    MaterialSample sample;
    bool ok = material_instance.sample(si, sg, sample);
    if (!ok)
        sample.pdf = 0;
    return sample;
}

public float material_eval_pdf(IMaterial material, SurfaceInteraction si, float3 wo)
{
    let lod_sampler = ExplicitLodSampler(0);
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);
    return material_instance.eval_pdf(si, wo);
}

public MaterialProperties material_collect_properties(IMaterial material, SurfaceInteraction si)
{
    let lod_sampler = ExplicitLodSampler(0);
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);
    return material_instance.collect_properties(si);
}

public MaterialProperties material_collect_properties_with_hints(IMaterial material, SurfaceInteraction si, uint hints)
{
    let lod_sampler = ExplicitLodSampler(0);
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints(hints));
    return material_instance.collect_properties(si);
}

public float3 material_eval_emission(IMaterial material, SurfaceInteraction si, uint hints)
{
    let lod_sampler = ExplicitLodSampler(0);
    return material.eval_emission(si, lod_sampler, MaterialInstanceHints(hints));
}

public float3 material_eval_emissive_radiance(IMaterial material, SurfaceInteraction si, uint hints)
{
    let lod_sampler = ExplicitLodSampler(0);
    return material.eval_emissive_radiance(si, lod_sampler, MaterialInstanceHints(hints));
}

public ExtraBsdfProperties material_collect_extra_bsdf_properties(IMaterial material, SurfaceInteraction si, float3 wo)
{
    let lod_sampler = ExplicitLodSampler(0);
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);
    ExtraBsdfProperties result = {};
    let maybe_result = material_instance.collect_extra_bsdf_properties(si, wo);
    if (maybe_result.hasValue) {
        result = maybe_result.value;
    }
    return result;
}

// _simple variants: work on individual materials via get_this()

public float3 material_eval_simple<
    SampleGenerator : ISampleGenerator>(uint3 seeds, IMaterial material, float2 uv, float3 wi, float3 wo)
{
    return material_eval<SampleGenerator>(seeds, material, setup_surface_interaction(uv, wi), wo);
}

public MaterialSample
material_sample_simple<SampleGenerator : ISampleGenerator>(uint3 seeds, IMaterial material, float2 uv, float3 wi)
{
    return material_sample<SampleGenerator>(seeds, material, setup_surface_interaction(uv, wi));
}

public float material_eval_pdf_simple(IMaterial material, float2 uv, float3 wi, float3 wo)
{
    return material_eval_pdf(material, setup_surface_interaction(uv, wi), wo);
}

public MaterialProperties material_collect_properties_simple(IMaterial material, float2 uv, float3 wi)
{
    return material_collect_properties(material, setup_surface_interaction(uv, wi));
}

public MaterialProperties
material_collect_properties_simple_with_hints(IMaterial material, float2 uv, float3 wi, uint hints)
{
    return material_collect_properties_with_hints(material, setup_surface_interaction(uv, wi), hints);
}

public float3 material_eval_emission_simple(IMaterial material, float2 uv, float3 wi, uint hints)
{
    return material_eval_emission(material, setup_surface_interaction(uv, wi), hints);
}

public float3 material_eval_emissive_radiance_simple(IMaterial material, float2 uv, float3 wi, uint hints)
{
    return material_eval_emissive_radiance(material, setup_surface_interaction(uv, wi), hints);
}

public ExtraBsdfProperties
material_collect_extra_bsdf_properties_simple(IMaterial material, float2 uv, float3 wi, float3 wo)
{
    return material_collect_extra_bsdf_properties(material, setup_surface_interaction(uv, wi), wo);
}

public uint material_get_lobe_types_simple<SampleGenerator : ISampleGenerator>(IMaterial material, float2 uv, float3 wi)
{
    let si = setup_surface_interaction(uv, wi);
    let lod_sampler = ExplicitLodSampler(0);
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);
    return uint(material_instance.get_lobe_types(si));
}

// _system variants: work on batches of materials via material_system

public float3 material_eval_system<
    SampleGenerator : ISampleGenerator>(uint3 seeds, uint32_t material_id, float2 uv, float3 wi, float3 wo)
{
    // material_id must be within buffer boundaries
    let material = g_scene.get_material(MaterialID(material_id));
    return material_eval<SampleGenerator>(seeds, material, setup_surface_interaction(uv, wi), wo);
}

public MaterialSample
material_sample_system<SampleGenerator : ISampleGenerator>(uint3 seeds, uint32_t material_id, float2 uv, float3 wi)
{
    // material_id must be within buffer boundaries
    let material = g_scene.get_material(MaterialID(material_id));
    return material_sample<SampleGenerator>(seeds, material, setup_surface_interaction(uv, wi));
}

public float material_eval_pdf_system(uint32_t material_id, float2 uv, float3 wi, float3 wo)
{
    // material_id must be within buffer boundaries
    let material = g_scene.get_material(MaterialID(material_id));
    return material_eval_pdf(material, setup_surface_interaction(uv, wi), wo);
}

public MaterialProperties material_collect_properties_system(uint32_t material_id, float2 uv, float3 wi)
{
    // material_id must be within buffer boundaries
    let material = g_scene.get_material(MaterialID(material_id));
    return material_collect_properties(material, setup_surface_interaction(uv, wi));
}

public ExtraBsdfProperties
material_collect_extra_bsdf_properties_system(uint32_t material_id, float2 uv, float3 wi, float3 wo)
{
    // material_id must be within buffer boundaries
    let material = g_scene.get_material(MaterialID(material_id));
    return material_collect_extra_bsdf_properties(material, setup_surface_interaction(uv, wi), wo);
}

public uint
material_get_lobe_types_system<SampleGenerator : ISampleGenerator>(uint32_t material_id, float2 uv, float3 wi)
{
    let si = setup_surface_interaction(uv, wi);
    let lod_sampler = ExplicitLodSampler(0);
    // material_id must be within buffer boundaries
    let material = g_scene.get_material(MaterialID(material_id));
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);
    return uint(material_instance.get_lobe_types(si));
}
