# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers


def test_pipeline_is_reflected_and_owns_its_settings() -> None:
    from falcor2.rendernodes.pathtracer_pipeline_node import AccumulationMode, PathTracerPipeline

    assert list(AccumulationMode.__members__) == ["off", "normal", "weighted"]

    pipeline = PathTracerPipeline.__new__(PathTracerPipeline)
    pipeline.optix_denoiser = SimpleNamespace(is_supported=True)
    pipeline._accumulation_mode = AccumulationMode.weighted
    pipeline._enable_dlss_rr = False
    pipeline._enable_optix_denoiser = False
    pipeline._enable_nan_overlay = False
    pipeline._accumulate_nan_count = True
    pipeline._pending_reset = False

    assert [info.name for info in pipeline._reflected_properties] == [
        "accumulation_mode",
        "enable_dlss_rr",
        "enable_optix_denoiser",
        "tone_map",
        "enable_nan_overlay",
        "accumulate_nan_count",
    ]

    pipeline.accumulation_mode = AccumulationMode.normal
    pipeline.enable_dlss_rr = True
    pipeline.enable_optix_denoiser = True
    pipeline.enable_nan_overlay = True
    pipeline.accumulate_nan_count = False

    assert pipeline.accumulation_mode == AccumulationMode.normal
    assert pipeline.enable_dlss_rr
    assert pipeline.enable_optix_denoiser
    assert pipeline.enable_nan_overlay
    assert not pipeline.accumulate_nan_count
    assert pipeline._pending_reset

    accumulator = pipeline._reflected_properties[0]
    optix = pipeline._reflected_properties[2]
    nan_accumulation = pipeline._reflected_properties[5]
    assert accumulator.ui_enable_if is not None
    assert not accumulator.ui_enable_if(pipeline)
    assert optix.ui_enable_if is not None
    assert optix.ui_enable_if(pipeline)
    assert nan_accumulation.ui_enable_if is not None
    assert nan_accumulation.ui_enable_if(pipeline)
    assert not hasattr(pipeline, "settings")
    assert not hasattr(pipeline, "persist_settings")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_single_iteration(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """PathTracerPipeline.__call__ returns a rendered image with correct dimensions."""
    from falcor2.rendernodes import AccumulationMode, PathTracerPipeline

    pipeline = PathTracerPipeline.create(device)
    assert pipeline.accumulation_mode == AccumulationMode.normal
    cam = helpers.create_test_camera(helmet_scene, width=64, height=64, fov_y=45)
    image = pipeline(helmet_scene, cam)
    assert image.width == 64
    assert image.height == 64
    data = image.to_numpy()
    assert not np.isnan(data).any()
    assert pipeline.accumulator._sample_count == 1
    assert pipeline.reweighting_accumulator._sample_count == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_accumulation_modes_select_exactly_one_policy(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    from falcor2.rendernodes import AccumulationMode, PathTracerPipeline

    camera = helpers.create_test_camera(helmet_scene, width=16, height=8, fov_y=45)
    pipeline = PathTracerPipeline.create(device)
    pipeline.tone_map = False
    pipeline.spp = 3

    pipeline.accumulation_mode = AccumulationMode.off
    pipeline(helmet_scene, camera)
    assert pipeline._iteration == 1
    assert pipeline.accumulator._sample_count == 0
    assert pipeline.reweighting_accumulator._sample_count == 0

    pipeline.accumulation_mode = AccumulationMode.normal
    pipeline(helmet_scene, camera)
    assert pipeline._iteration == 3
    assert pipeline.accumulator._sample_count == 3
    assert pipeline.reweighting_accumulator._sample_count == 0

    pipeline.accumulation_mode = AccumulationMode.weighted
    pipeline(helmet_scene, camera)
    assert pipeline._iteration == 3
    assert pipeline.accumulator._sample_count == 0
    assert pipeline.reweighting_accumulator._sample_count == 3


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_with_optix_denoiser(
    device_type: spy.DeviceType,
    device: spy.Device,
    helmet_scene: f2.Scene,
) -> None:
    from falcor2.rendernodes import PathTracerPipeline

    pipeline = PathTracerPipeline.create(device)
    if not pipeline.optix_denoiser_available:
        pytest.skip("OptiX requires a CUDA or CUDA-interoperable device")

    pipeline.tone_map = False
    pipeline.enable_optix_denoiser = True
    pipeline.output_spec = f2.ContainerSpec.tensor(spy.Format.rgba32_float)
    pipeline.guide_output_specs = {
        name: f2.ContainerSpec.tensor(spy.Format.rgba32_float)
        for name in pipeline.optix_denoiser.GUIDE_NAMES
    }
    camera = helpers.create_test_camera(helmet_scene, width=16, height=8, fov_y=45)

    command_encoder = device.create_command_encoder()
    image, guides = pipeline(helmet_scene, camera, output_guides=True, cmd=command_encoder)
    device.submit_command_buffer(command_encoder.finish())

    for name in pipeline.optix_denoiser.GUIDE_NAMES:
        assert isinstance(guides[name], spy.Tensor)
    assert pipeline.optix_denoiser._denoiser_key == ((8, 16), True, True)
    assert isinstance(image, spy.Tensor)
    assert image.shape == (8, 16)
    assert pipeline.optix_denoiser._copy_func is None
    assert pipeline.optix_denoiser._input is None
    assert pipeline.optix_denoiser._output is None
    assert pipeline.optix_denoiser._albedo is None
    assert pipeline.optix_denoiser._normal is None
    # The recorded command must retain the denoiser until its callback retires.
    pipeline.optix_denoiser._denoiser = None
    device.wait()
    assert np.isfinite(image.to_numpy()).all()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_accumulation_converges(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """Multiple pipeline calls accumulate samples; iteration auto-increments."""
    from falcor2.rendernodes import PathTracerPipeline

    pipeline = PathTracerPipeline.create(device)
    cam = helpers.create_test_camera(helmet_scene, width=64, height=64, fov_y=45)
    pipeline(helmet_scene, cam)
    assert pipeline._iteration == 1
    for _ in range(8):
        pipeline(helmet_scene, cam)
    assert pipeline._iteration == 9
    img9 = pipeline(helmet_scene, cam).to_numpy()
    assert pipeline._iteration == 10
    assert not np.isnan(img9).any()
    assert img9.max() > 0.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_resets_on_camera_change(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """Changing camera resets accumulator and iteration counter."""
    from falcor2.rendernodes import PathTracerPipeline

    pipeline = PathTracerPipeline.create(device)
    cam = helpers.create_test_camera(helmet_scene, width=64, height=64, fov_y=45)
    pipeline(helmet_scene, cam)
    assert pipeline._iteration == 1
    t = f2.Transform()
    t.translation = spy.float3(5.0, 0.0, 0.0)
    cam.entity.transform = t
    pipeline(helmet_scene, cam)
    assert pipeline._iteration == 1  # reset to 0, then incremented to 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_tonemap(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """Tonemap produces values in [0, 1] range."""
    from falcor2.rendernodes import PathTracerPipeline

    pipeline = PathTracerPipeline.create(device)
    pipeline.tone_map = True
    cam = helpers.create_test_camera(helmet_scene, width=64, height=64, fov_y=45)
    image = pipeline(helmet_scene, cam).to_numpy()
    assert image.min() >= 0.0
    assert image.max() <= 1.0 + 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_auto_exposure_forwards_delta_time_and_resets_history(
    device_type: spy.DeviceType,
    device: spy.Device,
    helmet_scene: f2.Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline forwards frame time and explicit reset invalidates exposure history."""
    from falcor2.rendernodes import PathTracerPipeline

    pipeline = PathTracerPipeline.create(device)
    pipeline.tonemapper.auto_exposure = True
    observed_delta_times: list[float] = []
    original_exec = pipeline.tonemapper._exec

    def record_exec(
        input: object,
        cmd: spy.CommandEncoder | None = None,
        delta_time: float = 1.0 / 60.0,
    ) -> object:
        observed_delta_times.append(delta_time)
        return original_exec(input, cmd=cmd, delta_time=delta_time)

    monkeypatch.setattr(pipeline.tonemapper, "_exec", record_exec)
    cam = helpers.create_test_camera(helmet_scene, width=16, height=8, fov_y=45)

    image = pipeline(helmet_scene, cam, delta_time=0.125)

    assert observed_delta_times == [0.125]
    assert pipeline.tonemapper._exposure_history_valid
    assert np.isfinite(image.to_numpy()).all()

    pipeline.reset()
    assert not pipeline.tonemapper._exposure_history_valid


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_request_reset_resets_next_render(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """request_reset() forces the next render call to restart accumulation."""
    from falcor2.rendernodes import PathTracerPipeline

    pipeline = PathTracerPipeline.create(device)
    cam = helpers.create_test_camera(helmet_scene, width=64, height=64, fov_y=45)
    pipeline(helmet_scene, cam)
    pipeline(helmet_scene, cam)
    assert pipeline._iteration == 2

    pipeline.request_reset()
    pipeline(helmet_scene, cam)

    assert pipeline._iteration == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_reweighting_mode_accumulates_each_spp_and_resolves_once(
    device_type: spy.DeviceType,
    device: spy.Device,
    helmet_scene: f2.Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from falcor2.rendernodes import (
        AccumulationMode,
        ReweightingAccumulatorOutput,
        PathTracerPipeline,
    )

    pipeline = PathTracerPipeline.create(device)
    pipeline.accumulation_mode = AccumulationMode.weighted
    pipeline.reweighting_accumulator.output_mode = ReweightingAccumulatorOutput.unbiased
    pipeline.spp = 3
    pipeline.tone_map = True
    camera = helpers.create_test_camera(helmet_scene, width=16, height=8, fov_y=45)
    events: list[str] = []
    accumulate_count = 0
    resolve_count = 0
    original_accumulate = pipeline.reweighting_accumulator.accumulate
    original_resolve = pipeline.reweighting_accumulator.resolve
    original_tonemap = pipeline.tonemapper._exec

    def record_accumulate(input: Any, cmd: spy.CommandEncoder | None = None) -> None:
        nonlocal accumulate_count
        accumulate_count += 1
        original_accumulate(input, cmd=cmd)

    def record_resolve(
        output_like: Any | None = None,
        cmd: spy.CommandEncoder | None = None,
        output_mode: ReweightingAccumulatorOutput | None = None,
    ) -> Any:
        nonlocal resolve_count
        resolve_count += 1
        events.append("resolve")
        return original_resolve(output_like=output_like, cmd=cmd, output_mode=output_mode)

    def record_tonemap(
        input: Any,
        cmd: spy.CommandEncoder | None = None,
        delta_time: float = 1.0 / 60.0,
    ) -> Any:
        events.append("tonemap")
        return original_tonemap(input, cmd=cmd, delta_time=delta_time)

    monkeypatch.setattr(pipeline.reweighting_accumulator, "accumulate", record_accumulate)
    monkeypatch.setattr(pipeline.reweighting_accumulator, "resolve", record_resolve)
    monkeypatch.setattr(pipeline.tonemapper, "_exec", record_tonemap)

    image = pipeline(helmet_scene, camera)
    data = image.to_numpy()

    assert accumulate_count == 3
    assert resolve_count == 1
    assert events == ["resolve", "tonemap"]
    assert pipeline._iteration == 3
    assert pipeline.reweighting_accumulator._sample_count == 3
    assert np.isfinite(data).all()
    assert data.max() > 0.0
    assert data.min() >= 0.0
    assert data.max() <= 1.0 + 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_reweighting_reset_and_resolve_controls(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    from falcor2.rendernodes import (
        AccumulationMode,
        ReweightingAccumulatorOutput,
        PathTracerPipeline,
    )

    pipeline = PathTracerPipeline.create(device)
    pipeline.accumulation_mode = AccumulationMode.normal
    pipeline.tone_map = False
    camera = helpers.create_test_camera(helmet_scene, width=16, height=8, fov_y=45)
    pipeline(helmet_scene, camera)
    assert pipeline.accumulator._sample_count == 1

    pipeline.accumulation_mode = AccumulationMode.weighted
    pipeline.reweighting_accumulator.output_mode = ReweightingAccumulatorOutput.unbiased
    image = pipeline(helmet_scene, camera)
    assert pipeline._iteration == 1
    assert pipeline.accumulator._sample_count == 0
    assert pipeline.reweighting_accumulator._sample_count == 1
    assert np.isfinite(image.to_numpy()).all()

    history = pipeline.reweighting_accumulator._history
    pipeline.reweighting_accumulator.kappa = 8.0
    pipeline.reweighting_accumulator.kappa_min = 0.5
    pipeline.reweighting_accumulator.output_mode = ReweightingAccumulatorOutput.reweighted
    pipeline(helmet_scene, camera)
    assert pipeline._iteration == 2
    assert pipeline.reweighting_accumulator._sample_count == 2
    assert pipeline.reweighting_accumulator._history is history

    transform = f2.Transform()
    transform.translation = spy.float3(5.0, 0.0, 0.0)
    camera.entity.transform = transform
    pipeline(helmet_scene, camera)
    assert pipeline._iteration == 1
    assert pipeline.reweighting_accumulator._sample_count == 1

    pipeline.accumulation_mode = AccumulationMode.normal
    pipeline(helmet_scene, camera)
    assert pipeline._iteration == 1
    assert pipeline.accumulator._sample_count == 1
    assert pipeline.reweighting_accumulator._sample_count == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_reweighting_tensor_output(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    from falcor2.rendernodes import (
        AccumulationMode,
        ReweightingAccumulatorOutput,
        PathTracerPipeline,
    )

    pipeline = PathTracerPipeline.create(device)
    pipeline.output_spec = f2.ContainerSpec.tensor(format=spy.Format.rgba32_float)
    pipeline.accumulation_mode = AccumulationMode.weighted
    pipeline.reweighting_accumulator.output_mode = ReweightingAccumulatorOutput.unbiased
    pipeline.tone_map = False
    camera = helpers.create_test_camera(helmet_scene, width=16, height=8, fov_y=45)

    image = pipeline(helmet_scene, camera)

    assert isinstance(image, spy.Tensor)
    assert image.shape == (8, 16)
    assert image.dtype.full_name == "vector<float,4>"
    assert np.isfinite(image.to_numpy()).all()


def test_pipeline_can_output_torch_tensor() -> None:
    """Torch output spec returns a torch.Tensor and propagates back to the path tracer."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if spy.DeviceType.cuda not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("CUDA device not available")

    from falcor2 import ContainerSpec
    from falcor2.rendernodes import PathTracerPipeline

    device = helpers.get_torch_device(spy.DeviceType.cuda, use_cache=False)
    scene = f2.Scene.load(device, "data/assets/kronos/Avocado/glTF-Binary/Avocado.glb")
    scene.update()
    cam = helpers.create_test_camera(scene, width=16, height=8, fov_y=45)

    pipeline = PathTracerPipeline.create(device)
    pipeline.output_spec = ContainerSpec(
        container_type=torch.Tensor, format=spy.Format.rgba32_float
    )

    image = pipeline(scene, cam)

    assert isinstance(image, torch.Tensor)
    assert tuple(image.shape) == (4, 8, 16)
    assert image.dtype == torch.float32


def test_pipeline_can_mix_torch_color_and_tensor_diffuse_albedo_guide() -> None:
    """Guide specs can return render-layout tensors while color output is torch."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if spy.DeviceType.cuda not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("CUDA device not available")

    from falcor2 import ContainerSpec
    from falcor2.rendernodes import PathTracerPipeline

    device = helpers.get_torch_device(spy.DeviceType.cuda, use_cache=False)
    scene = f2.Scene.load(device, "data/assets/kronos/Avocado/glTF-Binary/Avocado.glb")
    scene.update()
    cam = helpers.create_test_camera(scene, width=16, height=8, fov_y=45)

    pipeline = PathTracerPipeline.create(device)
    pipeline.output_spec = ContainerSpec.torch(format=spy.Format.rgba32_float)
    pipeline.guide_output_specs = {
        "diffuse_albedo": ContainerSpec.tensor(format=spy.Format.rgba32_float)
    }

    image, guides = pipeline(scene, cam, output_guides=True)
    albedo = guides["diffuse_albedo"]

    assert isinstance(image, torch.Tensor)
    assert tuple(image.shape) == (4, 8, 16)
    assert image.dtype == torch.float32
    assert albedo is not None
    assert isinstance(albedo, spy.Tensor)
    assert tuple(albedo.shape) == (8, 16)
    assert albedo.dtype.full_name == "vector<float,4>"
    albedo_data = albedo.to_numpy()
    assert np.isfinite(albedo_data).all()

    albedo_chw = albedo.to_torch().permute(2, 0, 1).contiguous()
    assert isinstance(albedo_chw, torch.Tensor)
    assert tuple(albedo_chw.shape) == (4, 8, 16)
    assert albedo_chw.dtype == torch.float32
    assert torch.isfinite(albedo_chw).all()
