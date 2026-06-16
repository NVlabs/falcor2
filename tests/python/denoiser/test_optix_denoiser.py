# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Any
import slangpy as spy
import pytest
import numpy as np
import falcor2 as f2
import falcor2.testing.helpers as helpers
import falcor2.testing.mthelpers as mthelpers


def create_checkerboard_pattern(width: int, height: int, channels: int, checker_size: int = 8):
    """Create a checkerboard pattern."""
    data = np.zeros((height, width, channels), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            checker = ((x // checker_size) + (y // checker_size)) % 2 == 0
            value = 1.0 if checker else 0.0
            data[y, x, :] = value

    return data


def add_noise(data: Any, noise_strength: float = 0.9):
    """Add random noise to an image."""
    noisy_data = data.copy()
    height, width, channels = data.shape

    rand = np.random.default_rng(2132)

    # Add noise only to every other pixel (like C++ version)
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            noise = rand.uniform(-noise_strength, noise_strength, channels)
            noisy_data[y, x, :] = np.clip(noisy_data[y, x, :] + noise, 0.0, 1.0)

    return noisy_data


def calculate_mse(img1: Any, img2: Any) -> float:
    """Calculate mean squared error between two images."""
    return float(np.mean((img1 - img2) ** 2))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_denoiser(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, cuda_interop=True)

    desc = f2.OptixDenoiserDesc()
    desc.model_kind = f2.OptixModelKind.aov
    desc.alpha_mode = f2.OptixAlphaMode.copy
    desc.max_width = 512
    desc.max_height = 512
    denoiser = f2.OptixDenoiser(device, desc)
    assert denoiser.device == device


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("shared", [False, True], ids=["non-shared", "shared"])
def test_denoise_checkerboard(device_type: spy.DeviceType, shared: bool):
    """Test denoising a checkerboard pattern with added noise."""
    if shared and device_type == spy.DeviceType.cuda:
        pytest.skip("Shared flag is not relevant on CUDA device")

    device = helpers.get_device(device_type, cuda_interop=True)

    # Create denoiser
    width = 256
    height = 256
    channels = 3  # RGB
    desc = f2.OptixDenoiserDesc()
    desc.model_kind = f2.OptixModelKind.aov
    desc.alpha_mode = f2.OptixAlphaMode.copy
    desc.max_width = width
    desc.max_height = height
    denoiser = f2.OptixDenoiser(device, desc)
    assert denoiser is not None

    # Setup checkerboard and noisy images
    original_data = create_checkerboard_pattern(width, height, channels)
    noisy_data = add_noise(original_data)

    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    if shared:
        usage |= spy.BufferUsage.shared

    noisy_buffer = spy.Tensor.from_numpy(device, noisy_data, usage=usage)
    denoised_buffer = spy.Tensor.empty_like(noisy_buffer)

    # Denoise
    denoiser.denoise(
        input=noisy_buffer.storage,
        output=denoised_buffer.storage,
        width=width,
        height=height,
    )
    denoised_data = denoised_buffer.to_numpy()

    # Verify that denoising improved the image quality
    mse_noisy = calculate_mse(original_data, noisy_data)
    mse_denoised = calculate_mse(original_data, denoised_data)

    # Make sure we get less noisy!
    assert mse_denoised < mse_noisy
    assert mse_denoised < 0.8 * mse_noisy

    # Very rough recorded values for mse just to make sure nothing's gone crazy
    assert abs(mse_noisy - 0.03) < 0.01
    assert abs(mse_denoised - 0.01) < 0.01


def get_samples_and_resolution(device_type: spy.DeviceType):
    samples = 1024
    width = 1024
    height = 1024
    if device_type == spy.DeviceType.cuda:
        width = 256
        height = 256
    return samples, width, height


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_raytraced_image(device_type: spy.DeviceType):
    samples, width, height = get_samples_and_resolution(device_type)
    show_tev = False

    time.sleep(3)

    # Setup device/scene etc
    device = helpers.get_device(device_type, cuda_interop=True)
    test_file = helpers.PROJECT_ROOT / "data/assets/kronos/DamagedHelmet/glTF/DamagedHelmet.gltf"
    scene, renderer = mthelpers.create_scene_and_renderer(device, test_file)
    renderer.use_raytracing_pipeline = True
    camera = scene.create_camera(width, height, 45)
    camera.transform.pos = spy.float3(0, 0.2, 2)
    camera.transform.rot = spy.math.quat_from_look_at(
        -spy.math.normalize(camera.transform.pos), spy.float3(0, 1, 0)
    )

    # Render a high-ish quality reference, and a noisy image with guides
    hq_image = renderer.render(scene, camera, spp=samples)
    noisy_image = renderer.render(scene, camera, spp=8)
    guides = renderer.render_guides(scene, camera, spp=8)
    albedo = guides["albedo"]
    normal = guides["shading_normal"]

    # Extract data and calculate MSE for noisy image
    hq_data = hq_image.to_numpy()
    noisy_data = noisy_image.to_numpy()
    mse_noisy = calculate_mse(hq_data, noisy_data)
    if show_tev:
        spy.tev.show(spy.Bitmap(hq_data))
        spy.tev.show(spy.Bitmap(noisy_data))

    # Buffer for denoised result
    denoised_image = spy.Tensor.empty_like(noisy_image)

    # Denoise with no guides.
    if True:
        desc = f2.OptixDenoiserDesc()
        desc.model_kind = f2.OptixModelKind.aov
        desc.alpha_mode = f2.OptixAlphaMode.copy
        desc.max_width = camera.width
        desc.max_height = camera.height
        denoiser = f2.OptixDenoiser(device, desc)
        denoiser.denoise(
            input=noisy_image.storage,
            output=denoised_image.storage,
            width=camera.width,
            height=camera.height,
            format=f2.OptixPixelFormat.float4,
        )
        denoised_data = denoised_image.to_numpy()
        if show_tev:
            spy.tev.show(spy.Bitmap(denoised_data))
        mse_denoised = calculate_mse(hq_data, denoised_data)

    # Denoise with albedo guide
    if True:
        desc = f2.OptixDenoiserDesc()
        desc.model_kind = f2.OptixModelKind.aov
        desc.alpha_mode = f2.OptixAlphaMode.copy
        desc.max_width = camera.width
        desc.max_height = camera.height
        desc.albedo_guide_layer = True
        denoiser = f2.OptixDenoiser(device, desc)
        denoiser.denoise(
            input=noisy_image.storage,
            output=denoised_image.storage,
            albedo=albedo.storage,
            width=camera.width,
            height=camera.height,
            format=f2.OptixPixelFormat.float4,
        )
        if show_tev:
            spy.tev.show(spy.Bitmap(denoised_data))
        mse_albedo_denoised = calculate_mse(hq_data, denoised_data)

    # Denoise with albedo + normal guides
    if True:
        desc = f2.OptixDenoiserDesc()
        desc.model_kind = f2.OptixModelKind.aov
        desc.alpha_mode = f2.OptixAlphaMode.copy
        desc.max_width = camera.width
        desc.max_height = camera.height
        desc.albedo_guide_layer = True
        desc.normal_guide_layer = True
        denoiser = f2.OptixDenoiser(device, desc)
        denoiser.denoise(
            input=noisy_image.storage,
            output=denoised_image.storage,
            albedo=albedo.storage,
            normal=normal.storage,
            width=camera.width,
            height=camera.height,
            format=f2.OptixPixelFormat.float4,
        )
        denoised_data = denoised_image.to_numpy()
        if show_tev:
            spy.tev.show(spy.Bitmap(denoised_data))
        mse_albedonormal_denoised = calculate_mse(hq_data, denoised_data)

    # Make sure they all get less noisy. Note: MSE isn't a guide enough
    # metric to verify denoise with guides is better than without guides.
    assert mse_noisy > mse_denoised
    assert mse_noisy > mse_albedo_denoised
    assert mse_noisy > mse_albedonormal_denoised


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
@pytest.mark.parametrize("custom_stream", [False, True], ids=["defaultstream", "customstream"])
def test_denoise_torch(device_type: spy.DeviceType, custom_stream: bool):
    """Test denoising a checkerboard pattern with added noise using torch tensors."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    device = helpers.get_torch_device(device_type)

    # Create denoiser
    width = 256
    height = 256
    channels = 3  # RGB
    desc = f2.OptixDenoiserDesc()
    desc.model_kind = f2.OptixModelKind.aov
    desc.alpha_mode = f2.OptixAlphaMode.copy
    desc.max_width = width
    desc.max_height = height
    denoiser = f2.OptixDenoiser(device, desc)
    assert denoiser is not None

    # Setup checkerboard and noisy images
    original_data = create_checkerboard_pattern(width, height, channels)
    noisy_data = add_noise(original_data)

    noisy_tensor = torch.from_numpy(noisy_data).to("cuda").contiguous()
    denoised_tensor = torch.empty_like(noisy_tensor).contiguous()

    # Make sure torch is all done so we can mess with custom streams below
    torch.cuda.synchronize()

    # Setup denoiser layer
    input_image = f2.OptixImage2D()
    input_image.format = f2.OptixPixelFormat.float3
    input_image.width = width
    input_image.height = height
    input_image.address = noisy_tensor.data_ptr()
    output_image = f2.OptixImage2D()
    output_image.format = f2.OptixPixelFormat.float3
    output_image.width = width
    output_image.height = height
    output_image.address = denoised_tensor.data_ptr()
    layer0 = f2.OptixDenoiserLayer()
    layer0.input = input_image
    layer0.output = output_image
    layer0.type = f2.OptixDenoiserAOVType.beauty

    # Denoise with/without custom cuda stream
    if custom_stream:
        stream = torch.cuda.Stream()
        denoiser.denoise(
            params=f2.OptixDenoiserParams(),
            guide_layer=f2.OptixDenoiserGuideLayer(),
            layers=[layer0],
            cuda_stream=spy.NativeHandle.from_cuda_stream(stream.cuda_stream),
        )
        stream.synchronize()
    else:
        denoiser.denoise(
            params=f2.OptixDenoiserParams(),
            guide_layer=f2.OptixDenoiserGuideLayer(),
            layers=[layer0],
        )

    # Verify that denoising improved the image quality
    denoised_data = denoised_tensor.cpu().numpy()
    mse_noisy = calculate_mse(original_data, noisy_data)
    mse_denoised = calculate_mse(original_data, denoised_data)

    # Make sure we get less noisy!
    assert mse_denoised < mse_noisy
    assert mse_denoised < 0.8 * mse_noisy

    # Very rough recorded values for mse just to make sure nothing's gone crazy
    assert abs(mse_noisy - 0.03) < 0.01
    assert abs(mse_denoised - 0.01) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
