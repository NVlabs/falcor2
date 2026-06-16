# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pipeline_single_iteration(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """PathTracerPipeline.__call__ returns a rendered image with correct dimensions."""
    from falcor2.rendernodes import PathTracerPipeline

    pipeline = PathTracerPipeline.create(device)
    cam = helpers.create_test_camera(helmet_scene, width=64, height=64, fov_y=45)
    image = pipeline(helmet_scene, cam)
    assert image.width == 64
    assert image.height == 64
    data = image.to_numpy()
    assert not np.isnan(data).any()
    assert data.max() > 0.0


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
    scene = f2.Scene(device, "data/assets/kronos/DamagedHelmet/glTF/DamagedHelmet.gltf")
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
    scene = f2.Scene(device, "data/assets/kronos/DamagedHelmet/glTF/DamagedHelmet.gltf")
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
