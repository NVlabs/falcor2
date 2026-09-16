# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2.testing.helpers as helpers
from falcor2.rendernodes import AccumulatorNode, AccumulatorPrecision


def _image(device: spy.Device, value: float, dtype: Any = np.float32) -> spy.Tensor:
    data = np.full((1, 1, 4), value, dtype=dtype)
    data[..., 3] = 0
    tensor = spy.Tensor.empty(
        device,
        shape=(1, 1),
        dtype="half4" if np.dtype(dtype) == np.float16 else "float4",
    )
    tensor.copy_from_numpy(data)
    return tensor


def _rgb_image(
    device: spy.Device,
    value: tuple[float, float, float],
    dtype: Any = np.float32,
) -> spy.Tensor:
    data = np.zeros((1, 1, 4), dtype=dtype)
    data[..., :3] = value
    tensor = spy.Tensor.empty(
        device,
        shape=(1, 1),
        dtype="half4" if np.dtype(dtype) == np.float16 else "float4",
    )
    tensor.copy_from_numpy(data)
    return tensor


def _supported_precisions(device: spy.Device) -> list[AccumulatorPrecision]:
    precisions = [AccumulatorPrecision.single, AccumulatorPrecision.single_compensated]
    if device.has_feature(spy.Feature.double):
        precisions.append(AccumulatorPrecision.double)
    return precisions


def _rgb(output: Any) -> np.ndarray:
    return output.to_numpy()[..., :3]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_accumulator_update_and_output(device_type: spy.DeviceType, device: spy.Device) -> None:
    for precision in _supported_precisions(device):
        acc = AccumulatorNode.create(device, precision)
        output_a = acc(_image(device, 1.0))
        output_a_data = output_a.to_numpy()
        output_b = acc(_image(device, 3.0))

        np.testing.assert_allclose(output_a_data[..., :3], 1.0, atol=1e-6)
        np.testing.assert_allclose(output_a_data[..., 3], 1.0, atol=1e-6)
        np.testing.assert_allclose(_rgb(output_b), 2.0, atol=1e-6)
        np.testing.assert_allclose(output_b.to_numpy()[..., 3], 1.0, atol=1e-6)
        assert acc._sample_count == 2


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_accumulator_nonfinite_sanitization_is_optional(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    invalid = _rgb_image(device, (float("nan"), float("inf"), 1.0))

    for precision in _supported_precisions(device):
        sanitized = AccumulatorNode.create(device, precision)
        sanitized_output = sanitized(invalid)
        np.testing.assert_array_equal(_rgb(sanitized_output), 0.0)
        assert sanitized.sanitize_non_finite is True

        unsanitized = AccumulatorNode.create(
            device,
            precision,
            sanitize_non_finite=False,
        )
        unsanitized_output = _rgb(unsanitized(invalid))
        assert not np.isfinite(unsanitized_output).all()

        history = sanitized._history
        sanitized.sanitize_non_finite = False
        assert sanitized._sample_count == 0
        assert sanitized._history is history


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_accumulator_max_sample_luminance_clamps_before_accumulation(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    source = np.array((8.0, 4.0, 2.0), dtype=np.float32)
    expected_scale = 1.0 / np.dot(source, np.array((0.2126, 0.7152, 0.0722)))
    expected_clamped = np.broadcast_to(source * expected_scale, (1, 1, 3))
    expected_unclamped = np.broadcast_to(source, (1, 1, 3))

    for precision in _supported_precisions(device):
        accumulator = AccumulatorNode.create(
            device,
            precision,
            clamp_sample_luminance=True,
            max_sample_luminance=1.0,
        )
        output = accumulator(_rgb_image(device, tuple(source)))
        np.testing.assert_allclose(_rgb(output), expected_clamped, atol=1e-5)

        history = accumulator._history
        accumulator.max_sample_luminance = float("inf")
        assert accumulator._sample_count == 0
        assert accumulator._history is history
        np.testing.assert_allclose(
            _rgb(accumulator(_rgb_image(device, tuple(source)))),
            expected_unclamped,
        )

        accumulator.max_sample_luminance = 0.0
        np.testing.assert_array_equal(
            _rgb(accumulator(_rgb_image(device, tuple(source)))),
            0.0,
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_accumulator_max_sample_luminance_rejects_invalid_values(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    accumulator = AccumulatorNode.create(device)

    for value in (-1.0, float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="must be non-negative"):
            accumulator.max_sample_luminance = value


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_accumulator_cancellation(device_type: spy.DeviceType, device: spy.Device) -> None:
    expected = np.float32(1.0 / 3.0)
    for precision in _supported_precisions(device):
        acc = AccumulatorNode.create(device, precision)
        acc(_image(device, float(2**24)))
        acc(_image(device, 1.0))
        output = acc(_image(device, float(-(2**24))))

        if precision == AccumulatorPrecision.single:
            np.testing.assert_array_equal(_rgb(output), 0.0)
        else:
            np.testing.assert_allclose(
                _rgb(output), expected, rtol=0, atol=1e-6, err_msg=precision.name
            )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_accumulator_promotes_fp16_history(device_type: spy.DeviceType, device: spy.Device) -> None:
    for precision in _supported_precisions(device):
        acc = AccumulatorNode.create(device, precision)
        acc(_image(device, 32768.0, np.float16))
        for _ in range(32):
            output = acc(_image(device, 1.0, np.float16))

        assert output.dtype.full_name == "vector<half,4>"
        np.testing.assert_allclose(_rgb(output), 994.0, rtol=0, atol=0.25)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_accumulator_reset(device_type: spy.DeviceType, device: spy.Device) -> None:
    acc = AccumulatorNode.create(device)
    acc(_image(device, 1.0))
    history = acc._history

    acc.reset()
    output = acc(_image(device, 9.0))

    np.testing.assert_allclose(_rgb(output), 9.0, atol=1e-6)
    assert acc._sample_count == 1
    assert acc._history is history


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_accumulator_precision_changes_reset_state(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    acc = AccumulatorNode.create(device)
    acc(_image(device, 1.0))
    history = acc._history
    update_func = acc._update_func

    acc.precision = AccumulatorPrecision.single_compensated
    assert acc._sample_count == 1
    assert acc._history is history

    acc.precision = AccumulatorPrecision.single
    assert acc.precision == AccumulatorPrecision.single
    assert acc._update_func is not update_func
    assert acc._sample_count == 0
    assert acc._history is None

    acc(_image(device, 2.0))
    assert isinstance(acc._history, spy.Tensor)
    assert acc._history.dtype.full_name == "vector<float,4>"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_input_spec_change_resets_history(device_type: spy.DeviceType, device: spy.Device) -> None:
    acc = AccumulatorNode.create(device)
    first = _image(device, 1.0, np.float16)
    second = _image(device, 3.0, np.float32)
    acc(first)
    old_history = acc._history

    output = acc(second)

    np.testing.assert_allclose(_rgb(output), 3.0, atol=1e-6)
    assert acc._sample_count == 1
    assert acc._history is not old_history


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_typed_tensor_state_and_output_reuse(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    input_tensor = spy.Tensor.empty(device, shape=(2, 3), dtype="half4")
    input_tensor.copy_from_numpy(np.ones((2, 3, 4), dtype=np.float16))
    acc = AccumulatorNode.create(device)

    output = acc(input_tensor)
    history = acc._history
    output_again = acc(input_tensor)

    assert isinstance(history, spy.Tensor)
    assert output.dtype.full_name == "vector<half,4>"
    assert output.shape == input_tensor.shape
    assert history.dtype.full_name == "AccumHistory4"
    assert acc._history is history
    assert output_again is output


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_texture_state_formats(device_type: spy.DeviceType, device: spy.Device) -> None:
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba16_float,
        width=3,
        height=2,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    texture.copy_from_numpy(np.ones((2, 3, 4), dtype=np.float16))

    acc = AccumulatorNode.create(device)
    output = acc(texture)
    assert isinstance(output, spy.Texture)
    assert output.format == spy.Format.rgba16_float
    assert isinstance(acc._history, spy.Tensor)
    assert acc._history.shape == (2, 3)
    assert acc._history.dtype.full_name == "AccumHistory4"

    if device.has_feature(spy.Feature.double):
        acc.precision = AccumulatorPrecision.double
        acc(texture)
        assert isinstance(acc._history, spy.Tensor)
        assert acc._history.dtype.full_name == "vector<double,4>"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_encoder_captures_counts(device_type: spy.DeviceType, device: spy.Device) -> None:
    acc = AccumulatorNode.create(device)
    encoder = device.create_command_encoder()
    acc(_image(device, 1.0), cmd=encoder)
    acc(_image(device, 3.0), cmd=encoder)
    acc.reset(cmd=encoder)
    acc(_image(device, 9.0), cmd=encoder)
    output = acc(_image(device, 11.0), cmd=encoder)
    device.submit_command_buffer(encoder.finish())
    device.wait()

    np.testing.assert_allclose(_rgb(output), 10.0, atol=1e-6)
    assert acc._sample_count == 2


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_sample_count_overflow(device_type: spy.DeviceType, device: spy.Device) -> None:
    acc = AccumulatorNode.create(device)
    acc(_image(device, 1.0))
    acc._sample_count = 0xFFFFFFFE
    output = acc._output
    history = acc._history

    acc(_image(device, 1.0))
    assert acc._sample_count == 0xFFFFFFFF

    with pytest.raises(OverflowError, match="Accumulator sample count reached UINT32_MAX"):
        acc(_image(device, 1.0))

    assert acc._sample_count == 0xFFFFFFFF
    assert acc._output is output
    assert acc._history is history


def test_torch_layout_and_double_history() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or spy.DeviceType.cuda not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("CUDA is required for torch accumulator coverage")

    device = helpers.get_torch_device(spy.DeviceType.cuda, use_cache=False)
    input_tensor = torch.ones((4, 2, 3), dtype=torch.float16, device="cuda")
    acc = AccumulatorNode.create(device)
    output = acc(input_tensor)
    device.wait()

    assert isinstance(output, torch.Tensor)
    assert output.shape == input_tensor.shape
    assert output.dtype == torch.float16
    assert isinstance(acc._history, spy.Tensor)
    assert acc._history.shape == (2, 3)
    assert acc._history.dtype.full_name == "AccumHistory4"

    if device.has_feature(spy.Feature.double):
        acc.precision = AccumulatorPrecision.double
        acc(input_tensor)
        device.wait()
        assert isinstance(acc._history, spy.Tensor)
        assert acc._history.shape == (2, 3)
        assert acc._history.dtype.full_name == "vector<double,4>"
