// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/testing/shader_test.slangh"

import falcor2.render;
import falcor2.render.materials.mtlx.mtlx;
import falcor2.utils;

public struct InputObserverBSDF : IBSDF
{
    public float tag;
    public BSDFEval eval_with_pdf(
        const float3 wi,
        const float3 wo,
        const BSDFContext bc,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        return BSDFEval();
    }
    public bool sample(const float3 wi, const BSDFSampleInput input, const BSDFContext bc, out BSDFSample sample)
    {
        sample = {};
        // Deliberately encode the input instead of a physical direction, so nested
        // routing can be checked independently of a leaf's sampling algorithm.
        sample.wo = float3(input.u_lobe, input.u_dir);
        sample.weight = float3(tag, float(input.aux_seed & 0xffffu), float(input.aux_seed >> 16));
        sample.flags = BSDFFlags::delta_reflection;
        return true;
    }
    public AlbedoContributions eval_albedo(const float3 wi)
    {
        return AlbedoContributions(float3(0.5), float3(0.5), float3(0.0), float3(0.0));
    }
    public RoughnessInformation eval_roughness(const float3 wi) { return {}; }
    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.04);
        }
    }
}

TEST_CASE(nested_materialx_sampling_forwards_conditional_inputs)
{
    typealias Leaf = mtlx::MxLayerableBSDF<InputObserverBSDF>;
    typealias Pair = mtlx::MxMixBSDF<Leaf, Leaf>;
    typealias Tree = mtlx::MxMixBSDF<Leaf, Pair>;
    var a = Leaf(InputObserverBSDF(2.0));
    var b = Leaf(InputObserverBSDF(1.0));
    var c = Leaf(InputObserverBSDF(3.0));
    var tree = Tree(c, Pair(a, b, 0.5), 0.25);
    var framed = mtlx::MxFramedWeightedBSDF<Tree>(tree, float3(1.0), Frame::identity());
    var weighted = mtlx::MxWeightedBSDF<mtlx::MxFramedWeightedBSDF<Tree>>(framed, float3(1.0));
    const float selectors[5] = { 0.1875, 0.5625, 0.875, 0.375, 0.75 };
    const float residuals[5] = { 0.5, 0.5, 0.5, 0.0, 0.0 };
    const float tags[5] = { 1.0, 2.0, 3.0, 2.0, 3.0 };
    for (uint i = 0; i < 5; ++i) {
        const BSDFSampleInput input = { selectors[i], float2(0.2, 0.8), 0xdeadbeefu };
        BSDFSample sample;
        CHECK(weighted.sample(float3(0.0, 0.0, 1.0), input, BSDFContext(), sample));
        CHECK_EQ(sample.wo, float3(residuals[i], input.u_dir));
        // Composite delta weights include floating-point probability ratios.
        CHECK(all(abs(sample.weight - float3(tags[i], 0xbeef, 0xdead)) < 0.01));
        CHECK_EQ(sample.pdf, 0.0);
        CHECK(sample.has_flag(BSDFFlags::delta));
    }
    const BSDFSampleInput input = { 0.375, float2(0.2, 0.8), 0xdeadbeefu };
    BSDFSample sample;
    CHECK(Pair(a, b, 0.0).sample(float3(0.0, 0.0, 1.0), input, BSDFContext(), sample));
    CHECK_EQ(sample.weight.x, 1.0);
    CHECK_EQ(sample.wo, float3(input.u_lobe, input.u_dir));
    CHECK(Pair(a, b, 1.0).sample(float3(0.0, 0.0, 1.0), input, BSDFContext(), sample));
    CHECK_EQ(sample.weight.x, 2.0);
    CHECK_EQ(sample.wo, float3(input.u_lobe, input.u_dir));
}


TEST_CASE(sample_input_auxiliary_stream_depends_only_on_seed)
{
    const uint seeds[3] = { 0u, 0x12345678u, 0xffffffffu };
    for (uint i = 0; i < 3; ++i) {
        const BSDFSampleInput input = { 0.5, float2(0.25, 0.75), seeds[i] };
        var a = input.make_aux_rng();
        var replay = input.make_aux_rng();
        BSDFSampleInput changed_coordinates = input;
        changed_coordinates.u_lobe = 0.125;
        changed_coordinates.u_dir = float2(0.625, 0.375);
        var same_seed = changed_coordinates.make_aux_rng();
        BSDFSampleInput changed_seed = input;
        changed_seed.aux_seed = seeds[(i + 1) % 3];
        var other_seed = changed_seed.make_aux_rng();
        bool differs = false;
        for (uint j = 0; j < 8; ++j) {
            uint word = a.next_uint();
            CHECK_EQ(word, replay.next_uint());
            CHECK_EQ(word, same_seed.next_uint());
            differs = (word != other_seed.next_uint()) || differs;
        }
        CHECK(differs);
    }
}

TEST_CASE(direct_bsdf_samples_preserve_direction_mapping_and_lobe_hints)
{
    const BSDFSampleInput input = { 0.5, float2(0.2, 0.7), 0x12345678u };
    LambertDiffuseBRDF diffuse = LambertDiffuseBRDF(float3(0.5));
    BSDFSample sample;
    CHECK(diffuse.sample(float3(0.0, 0.0, 1.0), input, BSDFContext(), sample));
    CHECK_EQ(sample.wo, square_to_cosine_hemisphere_concentric(input.u_dir));
    PBRTDielectricBSDF glass = PBRTDielectricBSDF(AnisotropicGGX(float2(0.0)), 1.5);
    CHECK(glass.sample(float3(0.0, 0.0, 1.0), input, BSDFContext(1.0, 1.5, BSDFFlags::reflection), sample));
    CHECK(sample.has_flag(BSDFFlags::delta_reflection));
    CHECK_EQ(sample.pdf, 0.0);
    CHECK(glass.sample(float3(0.0, 0.0, 1.0), input, BSDFContext(1.0, 1.5, BSDFFlags::transmission), sample));
    CHECK(sample.has_flag(BSDFFlags::delta_transmission));
    CHECK_EQ(sample.pdf, 0.0);
}

public struct CountingSampleGenerator : ISampleGenerator
{
    public uint count;
    uint seed_word;

    public __init(uint sample_index, uint scrambling_seed = 0xdeadbeefu)
    {
        count = sample_index;
        seed_word = scrambling_seed;
    }
    public __init(uint2 pixel, uint sample_index)
    {
        count = sample_index;
        seed_word = 0xdeadbeefu;
    }

    [mutating]
    public uint next_uint()
    {
        // Distinct upper bits expose ordering; nonzero low bits expose lossy seed conversion.
        const uint words[5] = { 0x200000ffu, 0x400000a5u, 0xffffff5au, seed_word, 0x80000001u };
        return words[count++ % 5];
    }
}

TEST_CASE(sample_input_factories_map_fields_in_order)
{
    CountingSampleGenerator sg = CountingSampleGenerator(0);
    const LightSampleInput light = make_light_sample_input(sg);
    CHECK_EQ(light.u_select, 0.125);
    CHECK_EQ(light.u_light, float2(0.25, 0x1.fffffep-1));
    CHECK_EQ(sg.count, 3u);
    CHECK_EQ(sg.next_uint(), 0xdeadbeefu);

    sg = CountingSampleGenerator(0);
    const BSDFSampleInput bsdf = make_bsdf_sample_input(sg);
    CHECK_EQ(bsdf.u_lobe, 0.125);
    CHECK_EQ(bsdf.u_dir, float2(0.25, 0x1.fffffep-1));
    CHECK_EQ(bsdf.aux_seed, 0xdeadbeefu);
    CHECK_EQ(sg.count, 4u);
    CHECK_EQ(sg.next_uint(), 0x80000001u);

    sg = CountingSampleGenerator(0);
    const MaterialSampleInput material = make_material_sample_input(sg);
    CHECK_EQ(material.u_lobe, bsdf.u_lobe);
    CHECK_EQ(material.u_dir, bsdf.u_dir);
    CHECK_EQ(material.aux_seed, bsdf.aux_seed);
    CHECK_EQ(sg.count, 4u);
    CHECK_EQ(sg.next_uint(), 0x80000001u);

    sg = CountingSampleGenerator(0);
    const PhaseFunctionSampleInput phase = make_phase_function_sample_input(sg);
    CHECK_EQ(phase.u_lobe, 0.125);
    CHECK_EQ(phase.u_dir, float2(0.25, 0x1.fffffep-1));
    CHECK_EQ(sg.count, 3u);
    CHECK_EQ(sg.next_uint(), 0xdeadbeefu);

    sg = CountingSampleGenerator(0);
    const VolumeDistanceSampleInput distance = make_volume_distance_sample_input(sg);
    CHECK_EQ(distance.u_channel, 0.125);
    CHECK_EQ(distance.u_distance, 0.25);
    CHECK_EQ(sg.count, 2u);
    CHECK_EQ(sg.next_uint(), 0xffffff5au);
}

TEST_CASE(sample_input_factories_advance_even_when_fields_are_ignored)
{
    CountingSampleGenerator sg = CountingSampleGenerator(0);
    make_light_sample_input(sg);
    CHECK_EQ(sg.count, 3u);
    make_bsdf_sample_input(sg);
    CHECK_EQ(sg.count, 7u);
    make_material_sample_input(sg);
    CHECK_EQ(sg.count, 11u);
    make_phase_function_sample_input(sg);
    CHECK_EQ(sg.count, 14u);
    make_volume_distance_sample_input(sg);
    CHECK_EQ(sg.count, 16u);
    CHECK_EQ(sg.next_uint(), 0x400000a5u);
}

TEST_CASE(sample_selector_binary_residuals)
{
    float selector = 0.125;
    CHECK(choose_sample_branch(selector, 0.25));
    CHECK_EQ(selector, 0.5);
    selector = 0.625;
    CHECK_FALSE(choose_sample_branch(selector, 0.25));
    CHECK_EQ(selector, 0.5);
    selector = 0.25;
    CHECK_FALSE(choose_sample_branch(selector, 0.25));
    CHECK_EQ(selector, 0.0);
    selector = 0.0;
    CHECK(choose_sample_branch(selector, 0.25));
    CHECK_EQ(selector, 0.0);
}

TEST_CASE(sample_selector_deterministic_branches)
{
    const float selectors[3] = { 0.0, 0.375, 0x1.fffffep-1 };
    for (uint i = 0; i < 3; ++i) {
        float selector = selectors[i];
        CHECK_FALSE(choose_sample_branch(selector, 0.0));
        CHECK_EQ(selector, selectors[i]);
        CHECK(choose_sample_branch(selector, 1.0));
        CHECK_EQ(selector, selectors[i]);
        CHECK_FALSE(choose_sample_branch(selector, -0.5));
        CHECK_EQ(selector, selectors[i]);
        CHECK(choose_sample_branch(selector, 1.5));
        CHECK_EQ(selector, selectors[i]);
    }
}

TEST_CASE(sample_selector_n_way_residuals)
{
    // Three intervals [0, .25), [.25, .75), [.75, 1), with a zero-weight
    // interval at .25 skipped by the caller's strict CDF comparison.
    CHECK_EQ(remap_sample_selector(0.125, 0.0, 0.25), 0.5);
    CHECK_EQ(remap_sample_selector(0.375, 0.25, 0.5), 0.25);
    CHECK_EQ(remap_sample_selector(0.875, 0.75, 0.25), 0.5);
    CHECK_EQ(remap_sample_selector(0.25, 0.25, 0.5), 0.0);
    CHECK_EQ(remap_sample_selector(0.75, 0.75, 0.25), 0.0);

    // Reusing .625 at the child would choose the wrong branch; its residual is .5.
    float selector = 0.625;
    CHECK_FALSE(choose_sample_branch(selector, 0.25));
    CHECK(choose_sample_branch(selector, 0.5625));
    CHECK_APPROX(selector, 8.0 / 9.0, 1.0e-7);
}

TEST_CASE(sample_selector_rounding_stays_below_one)
{
    const float max_selector = 0x1.fffffep-1;
    float selector = max_selector;
    CHECK_FALSE(choose_sample_branch(selector, 0.25));
    CHECK(selector >= 0.0 && selector < 1.0);
    CHECK_EQ(remap_sample_selector(max_selector, 0.0, 1.0), max_selector);
    CHECK_EQ(remap_sample_selector(max_selector, 0.5, 0.5), 0x1.fffffcp-1);
    // A selected interval's computed residual may round to its upper endpoint.
    CHECK_EQ(remap_sample_selector(0.5, 0.25, 0.25), max_selector);
}

void check_phase_sample<P : IPhaseFunction>(P phase, PhaseFunctionSampleInput input)
{
    const float3 wi = normalize(float3(1.0, 2.0, 3.0));
    PhaseFunctionSample sample;
    CHECK(phase.sample(wi, input, sample));
    CHECK(all(isfinite(sample.wo_ws)));
    CHECK_APPROX(length(sample.wo_ws), 1.0, 1.0e-6);
    CHECK_APPROX(sample.pdf, phase.eval_pdf(wi, sample.wo_ws), 1.0e-6);
    CHECK_APPROX(sample.weight, phase.eval(wi, sample.wo_ws) / sample.pdf, 1.0e-6);
}

TEST_CASE(phase_functions_accept_explicit_input_endpoints)
{
    const PhaseFunctionSampleInput inputs[3]
        = { { 0.0, float2(0.0) }, { 0.25, float2(0.375, 0.625) }, { 0x1.fffffep-1, float2(0x1.fffffep-1) } };
    for (uint i = 0; i < 3; ++i) {
        check_phase_sample(IsotropicPhaseFunction(), inputs[i]);
        check_phase_sample(HenyeyGreensteinPhaseFunction(0.65), inputs[i]);
        check_phase_sample(DualHenyeyGreensteinPhaseFunction(float3(0.65), float3(-0.25), float3(0.25)), inputs[i]);
        PhaseFunctionSample sample;
        CHECK_FALSE(NullPhaseFunction().sample(float3(0.0, 0.0, 1.0), inputs[i], sample));
        CHECK_EQ(sample.pdf, 0.0);
        CHECK_EQ(sample.weight, float3(0.0));
    }
}

TEST_CASE(phase_functions_use_separate_lobe_and_direction_coordinates)
{
    const float3 wi = float3(0.0, 0.0, 1.0);
    const float weights[3] = { 0.0, 0.25, 1.0 };
    const float selectors[4] = { 0.0, 0.125, 0.25, 0x1.fffffep-1 };
    for (uint w = 0; w < 3; ++w) {
        const let phase = DualHenyeyGreensteinPhaseFunction(float3(0.65), float3(-0.25), float3(weights[w]));
        for (uint s = 0; s < 4; ++s) {
            const PhaseFunctionSampleInput input = { selectors[s], float2(0.375, 0.625) };
            PhaseFunctionSample sample, selected, isotropic;
            CHECK(phase.sample(wi, input, sample));
            const float expected_g = selectors[s] < weights[w] ? 0.65 : -0.25;
            CHECK(HenyeyGreensteinPhaseFunction(expected_g).sample(wi, input, selected));
            CHECK_APPROX(sample.wo_ws, selected.wo_ws, 1.0e-6);
            CHECK(IsotropicPhaseFunction().sample(wi, input, isotropic));
            CHECK_EQ(isotropic.wo_ws, square_to_uniform_sphere(input.u_dir));
        }
    }
}

TEST_CASE(constant_light_uses_explicit_direction_coordinates)
{
    LightHeader header;
    ConstantLight light = ConstantLight(header, float3(1.0, 2.0, 3.0));
    const LightSampleContext ctx = {};
    const float2 coordinates[3] = { float2(0.0), float2(0.375, 0.625), float2(0x1.fffffep-1) };
    for (uint i = 0; i < 3; ++i) {
        LightSample sample;
        CHECK(light.sample(ctx, coordinates[i], sample));
        CHECK_EQ(sample.dir_ws, square_to_uniform_sphere(coordinates[i]));
        CHECK_APPROX(sample.pdf, float(float.INV_FOUR_PI), 1.0e-6);
        CHECK_APPROX(sample.weight * sample.pdf, light.radiance, 1.0e-6);
    }
}
