# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
from falcor2.testing import helpers
from falcor2.rendernodes import (
    PathTracerPipeline,
    ReferencePathTracerNode,
    SampleGeneratorType,
    SchedulingMode,
    VisibilityRayMode,
    VisibilityRayPlacement,
)


@pytest.mark.nightly
def test_sampler_property_reflection():
    reflection = f2.reflection.PythonClassReflection(ReferencePathTracerNode)
    assert reflection.find_property("sample_generator") is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("sampler", list(SampleGeneratorType))
def test_sampler_render_and_ser(device_type, device, helmet_scene, sampler):
    light = helmet_scene.create_entity().create_component(f2.EnvMapLight)
    light.env_map_path = "data/assets/envmaps/aerodynamics_workshop_512.hdr"
    helmet_scene.update()
    camera = helpers.create_test_camera(
        helmet_scene,
        width=32,
        height=32,
        fov_y=45,
        position=spy.float3(0, 0.0314, 0.12),
        rotation=spy.math.quat_from_look_at(spy.float3(0, 0, -1), spy.float3(0, 1, 0)),
    )
    node = ReferencePathTracerNode.create(device)
    node.sample_generator = sampler
    node.pixel_filter = f2.BoxReconstructionFilter(width=0)
    node.visibility_ray_mode = VisibilityRayMode.trace_ray
    node.visibility_ray_placement = VisibilityRayPlacement.late
    color = spy.Tensor.empty(device, (32, 32), spy.float4)
    images = []
    for mode in (SchedulingMode.simple, SchedulingMode.ser):
        if mode == SchedulingMode.ser and not device.has_feature(
            spy.Feature.shader_execution_reordering
        ):
            continue
        node.scheduling_mode = mode
        node._render(helmet_scene, camera, color, iteration=19)
        image = color.to_numpy()
        assert np.isfinite(image).all() and image[..., :3].max() > 0
        images.append(image.copy())
    if len(images) == 2:
        np.testing.assert_allclose(images[0], images[1], rtol=1e-4, atol=1e-5)
    # A different sample point must change the path while remaining repeatable.
    node._render(helmet_scene, camera, color, iteration=20)
    assert not np.array_equal(images[-1], color.to_numpy())
    node._render(helmet_scene, camera, color, iteration=19)
    np.testing.assert_array_equal(images[-1], color.to_numpy())


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_sampler_switch_invalidates_pipeline(device_type, device, helmet_scene):
    pipeline = PathTracerPipeline.create(device)
    camera = helpers.create_test_camera(helmet_scene, width=16, height=16, fov_y=45)
    assert pipeline.path_tracer.sample_generator == SampleGeneratorType.uniform
    pipeline(helmet_scene, camera)
    pipeline(helmet_scene, camera)
    assert pipeline._iteration == 2
    for sampler in list(SampleGeneratorType)[1:]:
        pipeline.path_tracer.sample_generator = sampler
        assert pipeline.path_tracer._render_func is None
        pipeline(helmet_scene, camera)
        assert pipeline._iteration == 1


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_sampler_payload_layout(device_type, device):
    module = spy.Module(device.load_module("pathtracer/sample_generator_layout.slang"))
    for sampler in SampleGeneratorType:
        function = module.payload_size.prelude(
            f"export struct SampleGenerator : ISampleGenerator = {sampler.slang_type_name};\n"
            "export struct OpacityEvaluator : IOpacityEvaluator = MaterialOpacityEvaluator;"
        )
        size = function()
        # CUDA needs exactly 32 words, not merely <= the 128-byte pipeline capacity.
        expected = 120 if sampler == SampleGeneratorType.uniform else 116
        assert size == (128 if device_type == spy.DeviceType.cuda else expected)
