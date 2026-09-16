# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2.testing.helpers as helpers
from falcor2.rendernodes import (
    ReweightingAccumulatorNode,
    ReweightingAccumulatorOutput,
)


def _tensor(device: spy.Device, values: np.ndarray, dtype: Any = np.float32) -> spy.Tensor:
    data = np.asarray(values, dtype=dtype)
    if data.ndim != 3 or data.shape[-1] != 4:
        raise ValueError("values must have shape (H, W, 4)")
    result = spy.Tensor.empty(
        device,
        shape=data.shape[:2],
        dtype="half4" if data.dtype == np.float16 else "float4",
    )
    result.copy_from_numpy(data)
    return result


def _constant_image(
    device: spy.Device,
    value: float,
    *,
    shape: tuple[int, int] = (1, 1),
    dtype: Any = np.float32,
) -> spy.Tensor:
    data = np.full((*shape, 4), value, dtype=dtype)
    data[..., 3] = 0.0
    return _tensor(device, data, dtype)


def _gpu_result(
    device: spy.Device,
    samples: np.ndarray,
) -> tuple[ReweightingAccumulatorNode, np.ndarray, np.ndarray]:
    accumulator = ReweightingAccumulatorNode.create(device)
    for sample in samples:
        accumulator.accumulate(_tensor(device, sample))
    unbiased = accumulator.resolve(output_mode=ReweightingAccumulatorOutput.unbiased).to_numpy()
    reweighted = accumulator.resolve(output_mode=ReweightingAccumulatorOutput.reweighted).to_numpy()
    return accumulator, unbiased, reweighted


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fixed_anchors_split_and_unbiased_reconstruction(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    values = np.array((1.0, 8.0, 32.0, 64.0), dtype=np.float32)
    sample = np.zeros((1, len(values), 4), dtype=np.float32)
    sample[..., :3] = values[None, :, None]
    accumulator, unbiased, _ = _gpu_result(device, sample[None, ...])
    assert accumulator._history is not None
    history = accumulator._history.to_numpy()

    np.testing.assert_allclose(history[0, 0, 0, :3], 1.0, atol=1e-6)
    np.testing.assert_allclose(history[0, 1, 1, :3], 8.0, atol=1e-6)
    np.testing.assert_allclose(history[0, 2, 1, :3], 32.0 / 7.0, atol=1e-5)
    np.testing.assert_allclose(history[0, 2, 2, :3], 192.0 / 7.0, atol=1e-5)
    np.testing.assert_allclose(history[0, 3, 2, :3], 64.0, atol=1e-5)
    np.testing.assert_allclose(history[..., 3], 0.0, atol=0.0)
    np.testing.assert_allclose(unbiased[0, :, :3], sample[0, :, :3], atol=1e-5)
    np.testing.assert_allclose(unbiased[..., 3], 1.0, atol=0.0)

    split_count = (32.0 / 7.0) / 8.0 + (192.0 / 7.0) / 64.0
    assert split_count == pytest.approx(1.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_count_weight_singleton_and_repeated_acceptance(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    accumulator = ReweightingAccumulatorNode.create(device, kappa=4.0)
    accumulator.accumulate(_constant_image(device, 8.0))
    singleton = accumulator.resolve().to_numpy()
    np.testing.assert_allclose(singleton[..., :3], 8.0, atol=1e-5)

    accumulator.accumulate(_constant_image(device, 8.0))
    accumulator.accumulate(_constant_image(device, 8.0))
    weighted = accumulator.resolve().to_numpy()
    np.testing.assert_allclose(weighted[..., :3], 112.0 / 15.0, atol=1e-5)

    accumulator.kappa = 1.0
    accepted = accumulator.resolve().to_numpy()
    np.testing.assert_allclose(accepted[..., :3], 8.0, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_kappa_monotonically_suppresses_without_reset(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    accumulator = ReweightingAccumulatorNode.create(device)
    for _ in range(3):
        accumulator.accumulate(_constant_image(device, 8.0))
    history = accumulator._history
    sample_count = accumulator._sample_count

    values = []
    for kappa in (0.0, 1.0, 4.0, 8.0):
        accumulator.kappa = kappa
        values.append(float(accumulator.resolve().to_numpy()[0, 0, 0]))

    assert values[0] >= values[1] >= values[2]
    assert accumulator._history is history
    assert accumulator._sample_count == sample_count


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_bin_count_change_releases_incompatible_history(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    accumulator = ReweightingAccumulatorNode.create(device)
    accumulator.accumulate(_constant_image(device, 8.0))
    previous_history = accumulator._history

    accumulator.bin_count = 4

    assert accumulator.bin_count == 4
    assert accumulator._history is None
    assert accumulator._overflow_count is None
    assert accumulator._input_spec is None
    assert accumulator._sample_count == 0

    accumulator.accumulate(_constant_image(device, 8.0))
    assert accumulator._history is not previous_history
    assert accumulator._history is not None
    assert accumulator._history.shape[-1] == 5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_spatial_singleton_suppression_fades_in(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    accumulator = ReweightingAccumulatorNode.create(device)
    image = np.zeros((3, 3, 4), dtype=np.float32)
    image[1, 1, :3] = 8.0
    accumulator.accumulate(_tensor(device, image))
    accumulator.accumulate(_tensor(device, image))

    output = accumulator.resolve().to_numpy()
    np.testing.assert_allclose(output[1, 1, :3], 112.0 / 15.0, atol=1e-5)

    empty_image = np.zeros_like(image)
    for _ in range(14):
        accumulator.accumulate(_tensor(device, empty_image))
    fully_reweighted = accumulator.resolve().to_numpy()
    np.testing.assert_allclose(fully_reweighted[1, 1, :3], 0.0, atol=0.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_value_weight_recovers_supported_bright_sample(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    samples = np.ones((5, 1, 2, 4), dtype=np.float32)
    samples[..., 3] = 0.0
    samples[:, 0, 1, :3] = 64.0
    samples[-1, 0, 0, :3] = 64.0

    _, _, reweighted = _gpu_result(device, samples)

    # The current pixel sees the bright value only once, so its count weight is
    # zero. Four accepted unit samples establish R=4, giving value weight 4/64.
    # At five samples, the interactive transition is 4/15 of the way from the
    # unweighted mean of 13.6 to the fully reweighted value of 1.6.
    np.testing.assert_allclose(reweighted[0, 0, :3], 10.4, atol=2e-5)
    np.testing.assert_allclose(reweighted[0, 1, :3], 64.0, atol=2e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reset_and_default_nonfinite_sanitization(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    accumulator = ReweightingAccumulatorNode.create(device)
    accumulator.accumulate(_constant_image(device, 8.0))
    history = accumulator._history
    accumulator.reset()

    invalid = np.full((1, 1, 4), (math.inf, 1.0, 1.0, 0.0), dtype=np.float32)
    output = accumulator(_tensor(device, invalid))
    np.testing.assert_allclose(output.to_numpy()[..., :3], 0.0, atol=0.0)
    assert accumulator._history is history
    assert accumulator._sample_count == 1
    assert accumulator.sanitize_non_finite is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fp16_tensor_and_texture_outputs(device_type: spy.DeviceType, device: spy.Device) -> None:
    half_input = _constant_image(device, 8.0, shape=(2, 3), dtype=np.float16)
    accumulator = ReweightingAccumulatorNode.create(
        device, output_mode=ReweightingAccumulatorOutput.unbiased
    )
    tensor_output = accumulator(half_input)
    assert isinstance(tensor_output, spy.Tensor)
    assert tensor_output.dtype.full_name == "vector<half,4>"
    assert tensor_output.shape == (2, 3)
    assert accumulator._history is not None
    assert accumulator._history.dtype.full_name == "vector<float,4>"

    texture_format = (
        spy.Format.rgba32_float if device_type == spy.DeviceType.cuda else spy.Format.rgba16_float
    )
    texture_dtype = np.float32 if device_type == spy.DeviceType.cuda else np.float16
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=texture_format,
        width=3,
        height=2,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    texture.copy_from_numpy(np.full((2, 3, 4), 2.0, dtype=texture_dtype))
    texture_output = accumulator(texture)
    assert isinstance(texture_output, spy.Texture)
    assert texture_output.format == texture_format
    np.testing.assert_allclose(texture_output.to_numpy()[..., :3], 2.0, atol=2e-3)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_input_format_and_empty_resolve(device_type: spy.DeviceType, device: spy.Device) -> None:
    accumulator = ReweightingAccumulatorNode.create(device)
    integer_input = spy.Tensor.empty(device, shape=(1, 1), dtype="uint4")
    with pytest.raises(ValueError, match="floating-point RGBA"):
        accumulator.accumulate(integer_input)
    with pytest.raises(RuntimeError, match="before accumulating"):
        accumulator.resolve()


def test_torch_output() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or spy.DeviceType.cuda not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("CUDA is required for torch reweighting accumulator coverage")

    device = helpers.get_torch_device(spy.DeviceType.cuda, use_cache=False)
    input_tensor = torch.ones((4, 2, 3), dtype=torch.float16, device="cuda")
    input_tensor[3] = 0.0
    accumulator = ReweightingAccumulatorNode.create(
        device, output_mode=ReweightingAccumulatorOutput.unbiased
    )
    output = accumulator(input_tensor)
    device.wait()

    assert isinstance(output, torch.Tensor)
    assert output.shape == input_tensor.shape
    assert output.dtype == torch.float16
    assert accumulator._history is not None
    assert accumulator._history.shape == (2, 3, 9)
    assert accumulator._history.dtype.full_name == "vector<float,4>"
