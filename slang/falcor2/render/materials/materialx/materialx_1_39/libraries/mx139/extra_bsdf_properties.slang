// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mx139;

import falcor2.utils;
import falcor2.render;

namespace mx139 {

public struct MxExtraBsdfPropertiesContext
{
    // Incoming direction in the adjusted material shading frame.
    public float3 wi_adjusted;
    // Accumulated closure weight from the root to the current leaf.
    public float3 closure_weight;
    // Current leaf visit during recursive closure_tree reporting.
    public int leaf_visit_index;
    // Maps recursive leaf visit order to result BSDF slot indices.
    public int leaf_visit_to_bsdf_index[ExtraBsdfProperties::max_bsdf_count];

    public __init()
    {
        wi_adjusted = float3(0.0);
        closure_weight = float3(1.0);
        leaf_visit_index = 0;
        for (int i = 0; i < ExtraBsdfProperties::max_bsdf_count; ++i)
            leaf_visit_to_bsdf_index[i] = i;
    }
}

public interface IMxExtraBsdfPropertiesCollector : ILayerableBSDF
{
    public float3 collect_extra_bsdf_properties(
        inout ExtraBsdfProperties result,
        inout int reported_count,
        inout MxExtraBsdfPropertiesContext ctx
    );
};

// This recursive material-property traversal is only used by closure_tree.
// The flat bsdf_mix root emits its own leaf loop during codegen.
public interface IMxMaterialPropertiesCollector : ILayerableBSDF
{
    public float3 collect_material_properties(
        inout MaterialProperties result,
        inout float roughness_norm,
        inout float3 guide_normal,
        inout MxExtraBsdfPropertiesContext ctx
    );
};

public float3 mx_accumulate_material_properties_leaf<Leaf : ILayerableBSDF>(
    inout MaterialProperties result,
    inout float roughness_norm,
    inout float3 guide_normal,
    const Leaf leaf,
    const float3 leaf_wi,
    const float3 final_leaf_weight,
    const float3 leaf_normal
)
{
    AlbedoContributions albedo = leaf.eval_albedo(leaf_wi);
    RoughnessInformation roughness = leaf.eval_roughness(leaf_wi);
    float3 weighted_reflection_albedo = albedo.reflection * final_leaf_weight;
    float2 bsdf_roughness = roughness.roughness;
    float contribution_split = min(bsdf_roughness.x + bsdf_roughness.y, 1.0);
    float contribution_weight = luminance(weighted_reflection_albedo);

    result.diffuse_reflection_albedo += weighted_reflection_albedo * contribution_split;
    result.specular_reflection_albedo += weighted_reflection_albedo * (1.0 - contribution_split);
    result.roughness += contribution_weight * (bsdf_roughness.x + bsdf_roughness.y) * 0.5;
    roughness_norm += contribution_weight;
    guide_normal += contribution_weight * leaf_normal;

    return leaf.layer_opacity(leaf_wi);
}

public float3 mx_push_extra_closure_weight(inout MxExtraBsdfPropertiesContext ctx, const float3 scale)
{
    const float3 saved_closure_weight = ctx.closure_weight;
    ctx.closure_weight = saved_closure_weight * scale;
    return saved_closure_weight;
}

public void mx_pop_extra_closure_weight(inout MxExtraBsdfPropertiesContext ctx, const float3 saved_closure_weight)
{
    ctx.closure_weight = saved_closure_weight;
}

} // namespace mx139

// Diagnostic traversal for closure_tree extra BSDF reporting. Leaf rows are
// written through `ctx.leaf_visit_to_bsdf_index`, so graph order stays outside
// the BSDF type and this diagnostic does not extend the core ILayerableBSDF contract.
// FramedWeighted is the leaf reporting boundary; raw BSDF leaves do not need
// diagnostic conformances.

public extension<Inner> mx139::MxWeightedBSDF<Inner> : mx139::IMxExtraBsdfPropertiesCollector
    where Inner : mx139::ILayerableBSDF
    where Inner : mx139::IMxExtraBsdfPropertiesCollector
{
    public float3 collect_extra_bsdf_properties(
        inout ExtraBsdfProperties result,
        inout int reported_count,
        inout mx139::MxExtraBsdfPropertiesContext ctx
    )
    {
        const float3 saved_closure_weight = mx139::mx_push_extra_closure_weight(ctx, this.weight);
        const float3 inner_opacity = this.inner.collect_extra_bsdf_properties(result, reported_count, ctx);
        mx139::mx_pop_extra_closure_weight(ctx, saved_closure_weight);
        return saturate(this.weight * inner_opacity);
    }
}

public extension<Inner> mx139::MxFramedWeightedBSDF<Inner> : mx139::IMxExtraBsdfPropertiesCollector
    where Inner : mx139::ILayerableBSDF
{
    public float3 collect_extra_bsdf_properties(
        inout ExtraBsdfProperties result,
        inout int reported_count,
        inout mx139::MxExtraBsdfPropertiesContext ctx
    )
    {
        const int leaf_visit_index = ctx.leaf_visit_index++;
        const int bsdf_index = leaf_visit_index < ExtraBsdfProperties::max_bsdf_count
            ? ctx.leaf_visit_to_bsdf_index[leaf_visit_index]
            : leaf_visit_index;
        const float3 local_wi = this.frame.to_local(ctx.wi_adjusted);
        reported_count = max(reported_count, bsdf_index + 1);
        if (bsdf_index >= ExtraBsdfProperties::max_bsdf_count)
            return saturate(this.weight * this.inner.layer_opacity(local_wi));

        AlbedoContributions albedo = this.inner.eval_albedo(local_wi);
        RoughnessInformation roughness = this.inner.eval_roughness(local_wi);
        result.bsdf_N[bsdf_index] = this.frame.normal;
        result.bsdf_T[bsdf_index] = this.frame.tangent;
        result.bsdf_B[bsdf_index] = this.frame.bitangent;
        result.bsdf_albedo[bsdf_index] = albedo.reflection + this.material_transmission * albedo.transmission;
        result.bsdf_weight[bsdf_index] += ctx.closure_weight * this.weight;
        result.bsdf_roughness[bsdf_index] = roughness.roughness;
        result.bsdf_scratch[bsdf_index] = roughness.scratch;

        return saturate(this.weight * this.inner.layer_opacity(local_wi));
    }
}

public extension<Top, Base> mx139::MxLayerBSDF<Top, Base> : mx139::IMxExtraBsdfPropertiesCollector
    where Top : mx139::ILayerableBSDF
    where Base : mx139::ILayerableBSDF
    where Top : mx139::IMxExtraBsdfPropertiesCollector
    where Base : mx139::IMxExtraBsdfPropertiesCollector
{
    public float3 collect_extra_bsdf_properties(
        inout ExtraBsdfProperties result,
        inout int reported_count,
        inout mx139::MxExtraBsdfPropertiesContext ctx
    )
    {
        const float3 top_opacity = this.top.collect_extra_bsdf_properties(result, reported_count, ctx);
        const float3 saved_closure_weight
            = mx139::mx_push_extra_closure_weight(ctx, saturate(float3(1.0) - top_opacity));
        const float3 base_opacity = this.base.collect_extra_bsdf_properties(result, reported_count, ctx);
        mx139::mx_pop_extra_closure_weight(ctx, saved_closure_weight);
        return base_opacity;
    }
}

public extension<FG, BG> mx139::MxMixBSDF<FG, BG> : mx139::IMxExtraBsdfPropertiesCollector
    where FG : mx139::ILayerableBSDF
    where BG : mx139::ILayerableBSDF
    where FG : mx139::IMxExtraBsdfPropertiesCollector
    where BG : mx139::IMxExtraBsdfPropertiesCollector
{
    public float3 collect_extra_bsdf_properties(
        inout ExtraBsdfProperties result,
        inout int reported_count,
        inout mx139::MxExtraBsdfPropertiesContext ctx
    )
    {
        const float3 bg_mix = float3(1.0) - this.mix_value;
        const float3 saved_closure_weight = mx139::mx_push_extra_closure_weight(ctx, bg_mix);
        const float3 bg_opacity = this.bg.collect_extra_bsdf_properties(result, reported_count, ctx);
        ctx.closure_weight = saved_closure_weight * this.mix_value;
        const float3 fg_opacity = this.fg.collect_extra_bsdf_properties(result, reported_count, ctx);
        mx139::mx_pop_extra_closure_weight(ctx, saved_closure_weight);
        return saturate(bg_mix * bg_opacity + this.mix_value * fg_opacity);
    }
}

public extension<In1, In2> mx139::MxAddBSDF<In1, In2> : mx139::IMxExtraBsdfPropertiesCollector
    where In1 : mx139::ILayerableBSDF
    where In2 : mx139::ILayerableBSDF
    where In1 : mx139::IMxExtraBsdfPropertiesCollector
    where In2 : mx139::IMxExtraBsdfPropertiesCollector
{
    public float3 collect_extra_bsdf_properties(
        inout ExtraBsdfProperties result,
        inout int reported_count,
        inout mx139::MxExtraBsdfPropertiesContext ctx
    )
    {
        const float3 opacity1 = this.in1.collect_extra_bsdf_properties(result, reported_count, ctx);
        const float3 opacity2 = this.in2.collect_extra_bsdf_properties(result, reported_count, ctx);
        return saturate(opacity1 + opacity2);
    }
}

public extension<Inner> mx139::MxWeightedBSDF<Inner> : mx139::IMxMaterialPropertiesCollector
    where Inner : mx139::ILayerableBSDF
    where Inner : mx139::IMxMaterialPropertiesCollector
{
    public float3 collect_material_properties(
        inout MaterialProperties result,
        inout float roughness_norm,
        inout float3 guide_normal,
        inout mx139::MxExtraBsdfPropertiesContext ctx
    )
    {
        const float3 saved_closure_weight = mx139::mx_push_extra_closure_weight(ctx, this.weight);
        const float3 inner_opacity = this.inner.collect_material_properties(result, roughness_norm, guide_normal, ctx);
        mx139::mx_pop_extra_closure_weight(ctx, saved_closure_weight);
        return saturate(this.weight * inner_opacity);
    }
}

public extension<Inner> mx139::MxFramedWeightedBSDF<Inner> : mx139::IMxMaterialPropertiesCollector
    where Inner : mx139::ILayerableBSDF
{
    public float3 collect_material_properties(
        inout MaterialProperties result,
        inout float roughness_norm,
        inout float3 guide_normal,
        inout mx139::MxExtraBsdfPropertiesContext ctx
    )
    {
        const float3 local_wi = this.frame.to_local(ctx.wi_adjusted);
        const float3 leaf_opacity = mx139::mx_accumulate_material_properties_leaf(
            result,
            roughness_norm,
            guide_normal,
            this.inner,
            local_wi,
            ctx.closure_weight * this.weight,
            this.frame.normal
        );
        return saturate(this.weight * leaf_opacity);
    }
}

public extension<Top, Base> mx139::MxLayerBSDF<Top, Base> : mx139::IMxMaterialPropertiesCollector
    where Top : mx139::ILayerableBSDF
    where Base : mx139::ILayerableBSDF
    where Top : mx139::IMxMaterialPropertiesCollector
    where Base : mx139::IMxMaterialPropertiesCollector
{
    public float3 collect_material_properties(
        inout MaterialProperties result,
        inout float roughness_norm,
        inout float3 guide_normal,
        inout mx139::MxExtraBsdfPropertiesContext ctx
    )
    {
        const float3 top_opacity = this.top.collect_material_properties(result, roughness_norm, guide_normal, ctx);
        const float3 saved_closure_weight
            = mx139::mx_push_extra_closure_weight(ctx, saturate(float3(1.0) - top_opacity));
        const float3 base_opacity = this.base.collect_material_properties(result, roughness_norm, guide_normal, ctx);
        mx139::mx_pop_extra_closure_weight(ctx, saved_closure_weight);
        return base_opacity;
    }
}

public extension<FG, BG> mx139::MxMixBSDF<FG, BG> : mx139::IMxMaterialPropertiesCollector
    where FG : mx139::ILayerableBSDF
    where BG : mx139::ILayerableBSDF
    where FG : mx139::IMxMaterialPropertiesCollector
    where BG : mx139::IMxMaterialPropertiesCollector
{
    public float3 collect_material_properties(
        inout MaterialProperties result,
        inout float roughness_norm,
        inout float3 guide_normal,
        inout mx139::MxExtraBsdfPropertiesContext ctx
    )
    {
        const float3 bg_mix = float3(1.0) - this.mix_value;
        const float3 saved_closure_weight = mx139::mx_push_extra_closure_weight(ctx, bg_mix);
        const float3 bg_opacity = this.bg.collect_material_properties(result, roughness_norm, guide_normal, ctx);
        ctx.closure_weight = saved_closure_weight * this.mix_value;
        const float3 fg_opacity = this.fg.collect_material_properties(result, roughness_norm, guide_normal, ctx);
        mx139::mx_pop_extra_closure_weight(ctx, saved_closure_weight);
        return saturate(bg_mix * bg_opacity + this.mix_value * fg_opacity);
    }
}

public extension<In1, In2> mx139::MxAddBSDF<In1, In2> : mx139::IMxMaterialPropertiesCollector
    where In1 : mx139::ILayerableBSDF
    where In2 : mx139::ILayerableBSDF
    where In1 : mx139::IMxMaterialPropertiesCollector
    where In2 : mx139::IMxMaterialPropertiesCollector
{
    public float3 collect_material_properties(
        inout MaterialProperties result,
        inout float roughness_norm,
        inout float3 guide_normal,
        inout mx139::MxExtraBsdfPropertiesContext ctx
    )
    {
        const float3 opacity1 = this.in1.collect_material_properties(result, roughness_norm, guide_normal, ctx);
        const float3 opacity2 = this.in2.collect_material_properties(result, roughness_norm, guide_normal, ctx);
        return saturate(opacity1 + opacity2);
    }
}
