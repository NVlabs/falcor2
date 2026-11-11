// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module test_lss_closest_hit;

__exported import falcor2.render;

public struct LSSHitMaterialInstance : IMaterialInstance
{
    public float3 emission;

    public BSDFEval eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both>(
        const SurfaceInteraction si,
        const float3 wo_ws,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        return {};
    }

    public bool sample(const SurfaceInteraction si, const MaterialSampleInput input, out MaterialSample sample)
    {
        sample = {};
        return false;
    }

    public override bool is_emissive() { return true; }
    public override float3 eval_emissive_radiance() { return emission; }
    public MaterialProperties collect_properties(const SurfaceInteraction si) { return {}; }
    public BSDFFlags get_lobe_types(const SurfaceInteraction si) { return BSDFFlags::none; }
};

public struct LSSHitMaterial : IMaterial
{
    typedef LSSHitMaterialInstance MaterialInstance;
    public MaterialHeader header;

    public override bool is_emissive() { return true; }

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};
        mi.emission = float3(si.uv, 1.0);
        if (let packed = si.hit_info.get()) {
            // Compare immediate shading against the persistent-hit fallback in the actual path tracer.
            let hit = packed.unpack();
            let fallback = g_scene.make_surface_interaction(hit, si.wi_ws);
            let previous = g_scene.get_previous_position_ws(LSSHitInfo(packed), si.position_ws);
            bool valid = all(abs(si.position_ws - fallback.position_ws) < 1.0e-6)
                && all(abs(si.normal_ws - fallback.normal_ws) < 1.0e-6) && all(abs(si.uv - fallback.uv) < 1.0e-6)
                && all(abs(si.shading_frame_ws.tangent - fallback.shading_frame_ws.tangent) < 1.0e-6)
                && all(abs(si.shading_frame_ws.bitangent - fallback.shading_frame_ws.bitangent) < 1.0e-6)
                && si.front_facing == fallback.front_facing && all(abs(previous - si.position_ws) < 1.0e-6);
            if (!valid)
                mi.emission = float3(1000.0);
        } else {
            mi.emission = float3(1000.0);
        }
        return mi;
    }
};
