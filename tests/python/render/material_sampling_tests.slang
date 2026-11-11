// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

__exported import scattering_tests;
import falcor2.tools.materials.scene_sampling_tools;

/// The shared kernels use world-space directions for this adapter.
public struct MaterialTarget : scattering_tests::IScatteringTestTarget
{
    public uint material_id;
    public float2 uv;
    public float3 wi;

    public bool sample(BSDFSampleInput input, out BSDFSample result)
    {
        const SurfaceInteraction si = setup_surface_interaction(uv, wi);
        let material = g_scene.get_material(MaterialID(material_id));
        let instance = material.setup_material_instance(si, ExplicitLodSampler(0), MaterialInstanceHints::none);
        MaterialSample sampled = {};
        const bool valid = instance.sample(si, input, sampled);
        result = {};
        result.wo = sampled.wo_ws;
        result.weight = sampled.weight;
        result.pdf = sampled.pdf;
        result.flags = sampled.flags;
        result.component = sampled.component;
        return valid;
    }


    public BSDFEval eval_with_pdf(float3 wo, BSDFComponent component = BSDFComponent::full_response)
    {
        const SurfaceInteraction si = setup_surface_interaction(uv, wi);
        let material = g_scene.get_material(MaterialID(material_id));
        let instance = material.setup_material_instance(si, ExplicitLodSampler(0), MaterialInstanceHints::none);
        return BSDFEval(instance.eval(si, wo, component), instance.eval_pdf(si, wo, component));
    }
};

public uint component_layout(MaterialTarget target)
{
    const SurfaceInteraction si = setup_surface_interaction(target.uv, target.wi);
    let material = g_scene.get_material(MaterialID(target.material_id));
    let instance = material.setup_material_instance(si, ExplicitLodSampler(0), MaterialInstanceHints::none);
    return instance.get_component_layout_id();
}

// Use a four-byte result: a scalar bool output can produce an undersized D3D12 structured buffer in SlangPy.
public uint component_rough(MaterialTarget target, uint component, float pdf, float threshold)
{
    const SurfaceInteraction si = setup_surface_interaction(target.uv, target.wi);
    let material = g_scene.get_material(MaterialID(target.material_id));
    let instance = material.setup_material_instance(si, ExplicitLodSampler(0), MaterialInstanceHints::none);
    return instance.is_component_rough(si, BSDFComponent(component), pdf, threshold) ? 1u : 0u;
}
