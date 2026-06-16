// SPDX-License-Identifier: Apache-2.0

implementing mx139;

import falcor2.utils;
import falcor2.render;

namespace mx139 {

public struct MxExtraBsdfPropertiesContext
{
    public float3 wi_node;
    public float3 eval_scale;
    public Frame node_frame_ws;
    public Frame diagnostic_frame_ws;
    public int visit_index;
    public int stable_indices[16];

    public __init()
    {
        wi_node = float3(0.0);
        eval_scale = float3(1.0);
        node_frame_ws = Frame::identity();
        diagnostic_frame_ws = Frame::identity();
        visit_index = 0;
        for (int i = 0; i < 16; ++i)
            stable_indices[i] = i;
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

public Frame mx_compose_child_frame_ws(const Frame parent_frame_ws, const Frame child_frame)
{
    Frame result;
    result.normal = normalize(parent_frame_ws.to_global(child_frame.normal));
    result.tangent = normalize(parent_frame_ws.to_global(child_frame.tangent));
    result.bitangent = normalize(parent_frame_ws.to_global(child_frame.bitangent));
    return result;
}

public float3 mx_append_extra_bsdf_leaf<Leaf : ILayerableBSDF>(
    inout ExtraBsdfProperties result,
    inout int reported_count,
    const Leaf leaf,
    const float3 leaf_wi,
    const float3 final_leaf_weight,
    const Frame leaf_frame_ws,
    const Frame diagnostic_frame_ws,
    const int stable_index
)
{
    result.bsdf_N[stable_index] = diagnostic_frame_ws.to_local(leaf_frame_ws.normal);
    result.bsdf_T[stable_index] = diagnostic_frame_ws.to_local(leaf_frame_ws.tangent);
    result.bsdf_B[stable_index] = diagnostic_frame_ws.to_local(leaf_frame_ws.bitangent);

    AlbedoContributions albedo = leaf.eval_albedo(leaf_wi);
    RoughnessInformation roughness = leaf.eval_roughness(leaf_wi);
    result.bsdf_albedo[stable_index] = albedo.reflection;
    result.bsdf_weight[stable_index] += final_leaf_weight;
    result.bsdf_roughness[stable_index] = roughness.roughness;
    result.bsdf_scratch[stable_index] = roughness.scratch;
    reported_count = max(reported_count, stable_index + 1);

    return leaf.layer_opacity(leaf_wi);
}

public float3 mx_push_extra_eval_scale(inout MxExtraBsdfPropertiesContext ctx, const float3 scale)
{
    const float3 saved_scale = ctx.eval_scale;
    ctx.eval_scale = saved_scale * scale;
    return saved_scale;
}

public void mx_pop_extra_eval_scale(inout MxExtraBsdfPropertiesContext ctx, const float3 saved_scale)
{
    ctx.eval_scale = saved_scale;
}

} // namespace mx139

// Diagnostic traversal for closure_tree extra BSDF reporting. Leaf rows are
// written through `ctx.stable_indices`, so graph order stays outside the BSDF
// type and this diagnostic does not extend the core ILayerableBSDF contract.
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
        const float3 saved_scale = mx139::mx_push_extra_eval_scale(ctx, this.weight);
        const float3 inner_opacity = this.inner.collect_extra_bsdf_properties(result, reported_count, ctx);
        mx139::mx_pop_extra_eval_scale(ctx, saved_scale);
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
        const int stable_index = ctx.stable_indices[ctx.visit_index++];
        const float3 local_wi = this.frame.to_local(ctx.wi_node);
        const Frame leaf_frame_ws = mx139::mx_compose_child_frame_ws(ctx.node_frame_ws, this.frame);
        const float3 leaf_opacity = mx139::mx_append_extra_bsdf_leaf(
            result,
            reported_count,
            this.inner,
            local_wi,
            ctx.eval_scale * this.weight,
            leaf_frame_ws,
            ctx.diagnostic_frame_ws,
            stable_index
        );
        return saturate(this.weight * leaf_opacity);
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
        const float3 saved_scale = mx139::mx_push_extra_eval_scale(ctx, saturate(float3(1.0) - top_opacity));
        const float3 base_opacity = this.base.collect_extra_bsdf_properties(result, reported_count, ctx);
        mx139::mx_pop_extra_eval_scale(ctx, saved_scale);
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
        const float3 saved_scale = mx139::mx_push_extra_eval_scale(ctx, bg_mix);
        const float3 bg_opacity = this.bg.collect_extra_bsdf_properties(result, reported_count, ctx);
        ctx.eval_scale = saved_scale * this.mix_value;
        const float3 fg_opacity = this.fg.collect_extra_bsdf_properties(result, reported_count, ctx);
        mx139::mx_pop_extra_eval_scale(ctx, saved_scale);
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
