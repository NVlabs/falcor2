# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy
import falcor2.testing.helpers as helpers

LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
HISTOGRAM_BIN_COUNT = 64
HISTOGRAM_WEIGHT_SCALE = 64
HISTOGRAM_LOG_MIN = -10.0
HISTOGRAM_LOG_MAX = 20.0
HISTOGRAM_LOW_PERCENT = 10.0
HISTOGRAM_HIGH_PERCENT = 90.0


def _luminance(color: np.ndarray) -> np.ndarray:
    sanitized_color = np.where(
        np.isfinite(color[..., :3]),
        np.maximum(color[..., :3], 0.0),
        0.0,
    )
    luminance = sanitized_color @ LUMINANCE_WEIGHTS
    return np.where(np.isfinite(luminance), luminance, 0.0)


def _luminance_histogram(
    color: np.ndarray,
    log_min: float = HISTOGRAM_LOG_MIN,
    log_max: float = HISTOGRAM_LOG_MAX,
    weight_scale: int = HISTOGRAM_WEIGHT_SCALE,
) -> np.ndarray:
    histogram = np.zeros(HISTOGRAM_BIN_COUNT, dtype=np.uint32)
    for luminance in _luminance(color).flat:
        if luminance <= 0.0:
            continue
        histogram_position = np.clip(
            (np.log2(luminance) - log_min) / (log_max - log_min),
            0.0,
            1.0,
        )
        bucket_position = histogram_position * (HISTOGRAM_BIN_COUNT - 1)
        bucket0 = min(int(bucket_position), HISTOGRAM_BIN_COUNT - 1)
        bucket1 = min(bucket0 + 1, HISTOGRAM_BIN_COUNT - 1)
        weight1 = int((bucket_position - np.floor(bucket_position)) * weight_scale + 0.5)
        weight0 = weight_scale - weight1
        if bucket0 != 0:
            histogram[bucket0] += weight0
        if bucket1 != 0:
            histogram[bucket1] += weight1
    return histogram


def _histogram_mean_log_luminance(
    histogram: np.ndarray,
    log_min: float = HISTOGRAM_LOG_MIN,
    log_max: float = HISTOGRAM_LOG_MAX,
    low_percent: float = HISTOGRAM_LOW_PERCENT,
    high_percent: float = HISTOGRAM_HIGH_PERCENT,
) -> float | None:
    total_weight = float(np.sum(histogram, dtype=np.uint64))
    if total_weight == 0.0:
        return None

    min_fraction_sum = total_weight * low_percent * 0.01
    max_fraction_sum = total_weight * high_percent * 0.01
    weighted_log_luminance = 0.0
    retained_weight = 0.0
    for index, bucket_weight in enumerate(histogram):
        local_weight = float(bucket_weight)
        discarded_weight = min(local_weight, min_fraction_sum)
        local_weight -= discarded_weight
        min_fraction_sum -= discarded_weight
        max_fraction_sum -= discarded_weight

        local_weight = min(local_weight, max(max_fraction_sum, 0.0))
        max_fraction_sum -= local_weight

        bucket_log_luminance = log_min + (log_max - log_min) * index / (HISTOGRAM_BIN_COUNT - 1)
        weighted_log_luminance += bucket_log_luminance * local_weight
        retained_weight += local_weight

    if retained_weight == 0.0:
        return None
    return weighted_log_luminance / retained_weight


def _histogram_target_ev(
    color: np.ndarray,
    exposure_key: float = 0.18,
    exposure_compensation: float = 0.0,
    min_exposure_ev: float = -16.0,
    max_exposure_ev: float = 16.0,
    log_min: float = HISTOGRAM_LOG_MIN,
    log_max: float = HISTOGRAM_LOG_MAX,
    low_percent: float = HISTOGRAM_LOW_PERCENT,
    high_percent: float = HISTOGRAM_HIGH_PERCENT,
) -> float:
    histogram = _luminance_histogram(color, log_min, log_max)
    mean_log_luminance = _histogram_mean_log_luminance(
        histogram,
        log_min,
        log_max,
        low_percent,
        high_percent,
    )
    if mean_log_luminance is None:
        return float(np.clip(exposure_compensation, min_exposure_ev, max_exposure_ev))
    return float(
        np.clip(
            np.log2(exposure_key) - mean_log_luminance + exposure_compensation,
            min_exposure_ev,
            max_exposure_ev,
        )
    )


def _tensor_from_numpy(device: spy.Device, data: np.ndarray) -> spy.Tensor:
    dtype = {3: spy.float3, 4: spy.float4}[data.shape[-1]]
    tensor = spy.Tensor.empty(device, data.shape[:-1], dtype)
    tensor.copy_from_numpy(data)
    return tensor


def _tonemap_with_shader_helper(
    device: spy.Device,
    operator_name: str,
    color: np.ndarray,
) -> np.ndarray:
    module = spy.Module(device.load_module("falcor2.utils"))
    tonemap = module.require_function(f"tonemap_{operator_name}")
    return tonemap(_tensor_from_numpy(device, color.astype(np.float32))).to_numpy()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
@pytest.mark.parametrize(
    "operator_name,color,expected",
    [
        (
            "aces_film",
            [0.0, 0.18, 1.0],
            [0.0, 0.2668989, 0.8037975],
        ),
        (
            "reinhard",
            [0.0, 1.0, 3.0],
            [0.0, 0.5, 0.75],
        ),
        (
            "clamp",
            [-1.0, 0.25, 2.0],
            [0.0, 0.25, 1.0],
        ),
    ],
)
def test_tonemap_shader_helpers_match_golden_values(
    device_type: spy.DeviceType,
    device: spy.Device,
    operator_name: str,
    color: list[float],
    expected: list[float],
) -> None:
    """Reusable tonemapping curves match fixed representative values."""
    color_data = np.array([[color]], dtype=np.float32)

    output = _tonemap_with_shader_helper(device, operator_name, color_data)

    assert np.allclose(output, np.array([[expected]], dtype=np.float32), atol=1e-6)


def _float_buffer_value(buffer: spy.Buffer) -> float:
    return float(buffer.to_numpy().view(np.float32)[0])


def test_histogram_weight_scale_prevents_uint32_overflow() -> None:
    from falcor2.rendernodes.tonemapper_node import (
        UINT32_MAX,
        _resolve_histogram_weight_scale,
    )

    assert _resolve_histogram_weight_scale(1920 * 1080) == HISTOGRAM_WEIGHT_SCALE

    pixel_count = 8192 * 8192
    weight_scale = _resolve_histogram_weight_scale(pixel_count)
    assert weight_scale == 63
    assert pixel_count * weight_scale <= UINT32_MAX
    assert pixel_count * (weight_scale + 1) > UINT32_MAX

    with pytest.raises(ValueError, match="too many pixels"):
        _resolve_histogram_weight_scale(UINT32_MAX + 1)


def _constant_tensor(
    device: spy.Device,
    value: float,
    shape: tuple[int, int] = (3, 5),
    alpha: float | None = None,
) -> spy.Tensor:
    channels = 3 if alpha is None else 4
    data = np.full((*shape, channels), value, dtype=np.float32)
    if alpha is not None:
        data[..., 3] = alpha
    return _tensor_from_numpy(device, data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemap_shader_histogram_matches_reference(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """The mapped histogram kernel matches the independent CPU reference."""
    module = spy.Module(device.load_module("falcor2.rendernodes.tonemapper"))
    color_data = np.array(
        [
            [[0.0, 0.0, 0.0], [0.25, 0.5, 1.0], [-1.0, 2.0, 0.5]],
            [[4.0, 2.0, 1.0], [1e-4, 2e-4, 4e-4], [np.nan, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    color = _tensor_from_numpy(device, color_data)
    histogram = spy.Tensor.from_numpy(device, np.zeros(HISTOGRAM_BIN_COUNT, dtype=np.uint32))

    module.tonemapper_accumulate_luminance_histogram(
        color=color,
        luminance_histogram=histogram.storage,
        histogram_weight_scale=HISTOGRAM_WEIGHT_SCALE,
        histogram_log_min=HISTOGRAM_LOG_MIN,
        histogram_log_max=HISTOGRAM_LOG_MAX,
    )

    assert np.array_equal(histogram.to_numpy(), _luminance_histogram(color_data))
    assert histogram.to_numpy()[0] == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_current_frame_auto_exposure_hits_current_target(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Current-frame mode meters and applies the input image in one call."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.current_frame
    input_data = np.full((3, 5, 3), 0.25, dtype=np.float32)
    input_tensor = _tensor_from_numpy(device, input_data)

    output = node(input_tensor).to_numpy()

    exposure_ev = _histogram_target_ev(input_data)
    expected = _tonemap_with_shader_helper(device, "aces_film", np.exp2(exposure_ev) * input_data)
    assert np.allclose(output, expected, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_black_borders_have_no_histogram_influence(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """The zero-weight black bucket makes exact-black borders irrelevant."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    base_data = np.full((4, 5, 3), 0.25, dtype=np.float32)
    bordered_data = np.zeros((6, 7, 3), dtype=np.float32)
    bordered_data[1:5, 1:6] = base_data
    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.current_frame

    base_output = node(_tensor_from_numpy(device, base_data)).to_numpy()
    assert node._luminance_histogram is not None
    base_histogram = node._luminance_histogram.to_numpy().copy()
    base_ev = _float_buffer_value(node._adapted_exposure_ev)

    bordered_output = node(_tensor_from_numpy(device, bordered_data)).to_numpy()

    assert np.array_equal(node._luminance_histogram.to_numpy(), base_histogram)
    assert _float_buffer_value(node._adapted_exposure_ev) == pytest.approx(base_ev)
    assert np.allclose(bordered_output[1:5, 1:6], base_output, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_histogram_percentiles_reject_luminance_outliers(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Ten-percent dark and bright tails leave the central luminance target."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    input_data = np.empty((10, 10, 3), dtype=np.float32)
    input_data.reshape(-1, 3)[:10] = np.exp2(-8.0)
    input_data.reshape(-1, 3)[10:90] = 0.25
    input_data.reshape(-1, 3)[90:] = np.exp2(12.0)
    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.current_frame

    output = node(_tensor_from_numpy(device, input_data)).to_numpy()

    expected_ev = np.log2(node.exposure_key / 0.25)
    assert node._adapted_exposure_ev is not None
    assert _float_buffer_value(node._adapted_exposure_ev) == pytest.approx(expected_ev, abs=2e-3)
    expected_middle = _tonemap_with_shader_helper(
        device, "aces_film", np.full((80, 3), 0.18, dtype=np.float32)
    )
    assert np.allclose(output.reshape(-1, 3)[10:90], expected_middle, atol=5e-4)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_all_black_fallback_is_neutral_and_retains_previous_frame_history(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """No valid samples initialize neutral EV and cannot disturb valid history."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    black = _constant_tensor(device, 0.0)
    lit = _constant_tensor(device, 0.25)
    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.current_frame
    node.exposure_compensation = 1.0

    assert np.array_equal(node(black).to_numpy(), np.zeros((3, 5, 3), dtype=np.float32))
    assert node._adapted_exposure_ev is not None
    assert _float_buffer_value(node._adapted_exposure_ev) == pytest.approx(1.0)

    node.exposure_compensation = 0.0
    node.auto_exposure_mode = AutoExposureMode.previous_frame
    node.reset()
    node(lit)
    lit_data = np.full((3, 5, 3), 0.25, dtype=np.float32)
    expected_ev = _histogram_target_ev(lit_data)
    assert _float_buffer_value(node._adapted_exposure_ev) == pytest.approx(expected_ev, abs=1e-5)

    node(black)
    assert _float_buffer_value(node._adapted_exposure_ev) == pytest.approx(expected_ev, abs=1e-5)

    output = node(lit).to_numpy()
    expected = _tonemap_with_shader_helper(device, "aces_film", np.exp2(expected_ev) * lit_data)
    assert np.allclose(output, expected, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "first_value,second_value,speed_name,speed,delta_time",
    [
        (0.25, 4.0, "adaptation_speed_down", 2.0, 0.25),
        (4.0, 0.25, "adaptation_speed_up", 1.0, 0.25),
    ],
)
def test_tonemapper_previous_frame_mode_clamps_ev_change(
    device_type: spy.DeviceType,
    device: spy.Device,
    first_value: float,
    second_value: float,
    speed_name: str,
    speed: float,
    delta_time: float,
) -> None:
    """Previous-frame mode uses prior EV and stores a bounded EV for the next frame."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.previous_frame
    setattr(node, speed_name, speed)
    first = _constant_tensor(device, first_value)
    second = _constant_tensor(device, second_value)

    first_output = node(first, delta_time=delta_time).to_numpy()
    second_output = node(second, delta_time=delta_time).to_numpy()
    third_output = node(second, delta_time=delta_time).to_numpy()

    first_data = np.full((3, 5, 3), first_value, dtype=np.float32)
    second_data = np.full((3, 5, 3), second_value, dtype=np.float32)
    first_ev = _histogram_target_ev(first_data)
    second_target_ev = _histogram_target_ev(second_data)
    max_change = speed * delta_time
    second_ev = first_ev + np.clip(second_target_ev - first_ev, -max_change, max_change)
    expected_first = _tonemap_with_shader_helper(
        device, "aces_film", first_data * np.exp2(first_ev)
    )
    expected_second = _tonemap_with_shader_helper(
        device,
        "aces_film",
        np.full_like(second_output, second_value * np.exp2(first_ev)),
    )
    expected_third = _tonemap_with_shader_helper(
        device,
        "aces_film",
        np.full_like(third_output, second_value * np.exp2(second_ev)),
    )

    assert np.allclose(first_output, expected_first, atol=1e-5)
    assert np.allclose(second_output, expected_second, atol=1e-5)
    assert np.allclose(third_output, expected_third, atol=1e-5)

    node.reset()
    reset_output = node(second, delta_time=delta_time).to_numpy()
    expected_reset = _tonemap_with_shader_helper(
        device, "aces_film", second_data * np.exp2(second_target_ev)
    )
    assert np.allclose(reset_output, expected_reset, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_runtime_bounds_reinitialize_clamped_history(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Narrowing the EV range immediately reinitializes history inside the new bounds."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    input_data = np.full((3, 5, 3), np.exp2(-5.0), dtype=np.float32)
    input_tensor = _tensor_from_numpy(device, input_data)
    node = TonemapperNode.create(device)
    node.auto_exposure_mode = AutoExposureMode.previous_frame

    node(input_tensor)
    assert node._adapted_exposure_ev is not None
    assert _float_buffer_value(node._adapted_exposure_ev) > 1.0

    node.max_exposure_ev = 1.0
    output = node(input_tensor).to_numpy()

    assert _float_buffer_value(node._adapted_exposure_ev) == pytest.approx(1.0)
    expected = _tonemap_with_shader_helper(device, "aces_film", np.exp2(1.0) * input_data)
    assert np.allclose(output, expected, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "operator_name",
    ["aces_film", "reinhard", "clamp"],
)
def test_tonemapper_selects_operator(
    device_type: spy.DeviceType,
    device: spy.Device,
    operator_name: str,
) -> None:
    """Each public operator applies manual exposure and preserves alpha."""
    from falcor2.rendernodes import TonemapperNode, TonemappingOperator

    operator = TonemappingOperator[operator_name]
    node = TonemapperNode.create(device)
    node.operator = operator
    node.auto_exposure = False
    node.exposure_compensation = 1.0
    input_data = np.array(
        [[[-0.25, 0.25, 2.0, 0.375], [0.5, 1.0, 4.0, 0.75]]],
        dtype=np.float32,
    )

    output = node(_tensor_from_numpy(device, input_data)).to_numpy()
    expected = _tonemap_with_shader_helper(device, operator_name, 2.0 * input_data[..., :3])

    assert node.operator == operator
    assert np.allclose(output[..., :3], expected, atol=1e-6)
    assert np.array_equal(output[..., 3], input_data[..., 3])
    assert node._luminance_histogram is None
    assert node._adapted_exposure_ev is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_operator_applies_to_auto_exposure_paths(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Operator selection is shared by immediate and fused auto-exposure paths."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode, TonemappingOperator

    input_data = np.full((3, 5, 3), 0.25, dtype=np.float32)
    input_tensor = _tensor_from_numpy(device, input_data)
    expected_ev = _histogram_target_ev(input_data)
    expected = _tonemap_with_shader_helper(device, "reinhard", np.exp2(expected_ev) * input_data)
    node = TonemapperNode.create(device, operator=TonemappingOperator.reinhard)
    node.auto_exposure_mode = AutoExposureMode.previous_frame

    immediate_output = node(input_tensor).to_numpy()
    fused_output = node(input_tensor).to_numpy()

    assert np.allclose(immediate_output, expected, atol=1e-5)
    assert np.allclose(fused_output, expected, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_nonfinite_metering_does_not_poison_exposure(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Invalid meter samples cannot make exposure or finite-pixel output non-finite."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    input_data = np.array(
        [
            [
                [0.25, 0.25, 0.25],
                [0.0, 0.0, 0.0],
                [-1.0, -2.0, -3.0],
                [np.nan, 1.0, 1.0],
                [np.inf, 1.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.current_frame

    output = node(_tensor_from_numpy(device, input_data)).to_numpy()

    assert np.isfinite(output[0, 0]).all()
    assert node._adapted_exposure_ev is not None
    assert np.isfinite(_float_buffer_value(node._adapted_exposure_ev))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_queued_previous_frame_sequence(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """A caller-owned encoder orders initialization, fused metering, and next-frame EV."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.previous_frame
    node.adaptation_speed_down = 2.0
    first_value = 0.25
    second_value = 4.0
    delta_time = 0.25
    first = _constant_tensor(device, first_value)
    second = _constant_tensor(device, second_value)
    encoder = device.create_command_encoder()

    node(first, cmd=encoder, delta_time=delta_time)
    node(second, cmd=encoder, delta_time=delta_time)
    output = node(second, cmd=encoder, delta_time=delta_time)
    device.submit_command_buffer(encoder.finish())
    device.wait()

    first_data = np.full((3, 5, 3), first_value, dtype=np.float32)
    first_ev = _histogram_target_ev(first_data)
    expected_ev = first_ev - node.adaptation_speed_down * delta_time
    expected = _tonemap_with_shader_helper(
        device,
        "aces_film",
        np.full((*second.shape.as_tuple(), 3), second_value * np.exp2(expected_ev)),
    )
    assert np.allclose(output.to_numpy(), expected, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_tonemapper_auto_exposure_histogram_is_fixed_size_and_reused(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """The same 64-bin histogram is cleared and reused for every image size."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.current_frame

    node(_constant_tensor(device, 0.25, shape=(4, 4)))
    original_histogram = node._luminance_histogram
    assert original_histogram is not None
    assert original_histogram.shape.as_tuple() == (HISTOGRAM_BIN_COUNT,)
    assert np.sum(original_histogram.to_numpy()) == 16 * HISTOGRAM_WEIGHT_SCALE

    node(_constant_tensor(device, 0.25, shape=(2, 3)))
    assert node._luminance_histogram is original_histogram
    assert np.sum(original_histogram.to_numpy()) == 6 * HISTOGRAM_WEIGHT_SCALE

    node(_constant_tensor(device, 0.25, shape=(5, 4)))
    assert node._luminance_histogram is original_histogram
    assert np.sum(original_histogram.to_numpy()) == 20 * HISTOGRAM_WEIGHT_SCALE


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_auto_exposure_texture_preserves_contract(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Texture auto-exposure preserves RGBA format, dimensions, and alpha."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    data = np.full((2, 3, 4), 0.25, dtype=np.float32)
    data[..., 3] = 0.375
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=3,
        height=2,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=data,
    )
    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.current_frame

    output = node(texture)
    output_data = output.to_numpy()

    assert isinstance(output, spy.Texture)
    assert output.format == texture.format
    assert (output.height, output.width) == (2, 3)
    exposure_ev = _histogram_target_ev(data[..., :3])
    expected = _tonemap_with_shader_helper(
        device, "aces_film", np.exp2(exposure_ev) * data[..., :3]
    )
    assert np.allclose(output_data[..., :3], expected, atol=1e-5)
    assert np.allclose(output_data[..., 3], 0.375)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_auto_exposure_half_tensor_preserves_contract(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Half RGBA tensors keep their dtype, dimensions, and alpha."""
    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    data = np.full((2, 3, 4), 0.25, dtype=np.float16)
    data[..., 3] = np.float16(0.375)
    input_tensor = spy.Tensor.empty(device, shape=(2, 3), dtype="half4")
    input_tensor.copy_from_numpy(data)
    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.current_frame

    output = node(input_tensor)
    output_data = output.to_numpy()

    assert output.dtype.full_name == "vector<half,4>"
    assert output.shape == input_tensor.shape
    rgb_data = data[..., :3].astype(np.float32)
    exposure_ev = _histogram_target_ev(rgb_data)
    expected = _tonemap_with_shader_helper(device, "aces_film", np.exp2(exposure_ev) * rgb_data)
    assert np.allclose(
        output_data[..., :3],
        expected,
        atol=1e-3,
    )
    assert np.allclose(output_data[..., 3], 0.375, atol=1e-3)

    node.auto_exposure_mode = AutoExposureMode.previous_frame
    node.reset()
    node(input_tensor)
    previous_frame_output = node(input_tensor).to_numpy()
    assert np.allclose(previous_frame_output, output_data, atol=1e-3)


def test_tonemapper_auto_exposure_torch_preserves_contract() -> None:
    """CUDA torch CHW input remains a torch tensor and meters in render layout."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if spy.DeviceType.cuda not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("CUDA device not available")

    from falcor2.rendernodes import AutoExposureMode, TonemapperNode

    device = helpers.get_torch_device(spy.DeviceType.cuda, use_cache=False)
    input_tensor = torch.full((4, 2, 3), 0.25, dtype=torch.float32, device="cuda")
    input_tensor[3].fill_(0.375)
    node = TonemapperNode.create(device)
    node.auto_exposure = True
    node.auto_exposure_mode = AutoExposureMode.current_frame

    output = node(input_tensor)
    output_data = output.detach().cpu().numpy()

    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (4, 2, 3)
    assert output.dtype == torch.float32
    rgb_data = np.full((2, 3, 3), 0.25, dtype=np.float32)
    exposure_ev = _histogram_target_ev(rgb_data)
    expected = _tonemap_with_shader_helper(device, "aces_film", np.exp2(exposure_ev) * rgb_data)
    assert np.allclose(
        output_data[:3],
        np.moveaxis(expected, -1, 0),
        atol=1e-5,
    )
    assert np.allclose(output_data[3], 0.375)

    with pytest.raises(ValueError, match="must not be empty"):
        node(torch.empty((4, 0, 2), dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_tonemapper_configuration_validation(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Invalid public settings and unsupported meter inputs fail clearly."""
    from falcor2.rendernodes import TonemapperNode

    node = TonemapperNode.create(device)
    with pytest.raises(TypeError, match="auto_exposure"):
        node.auto_exposure = 1  # type: ignore[assignment]
    with pytest.raises(ValueError, match="exposure_key"):
        node.exposure_key = 0.0
    with pytest.raises(ValueError, match="histogram_log_min"):
        node.histogram_log_min = node.histogram_log_max
    with pytest.raises(ValueError, match="histogram_log_max"):
        node.histogram_log_max = node.histogram_log_min
    with pytest.raises(ValueError, match="histogram_low_percent"):
        node.histogram_low_percent = -1.0
    with pytest.raises(ValueError, match="histogram_low_percent"):
        node.histogram_low_percent = 91.0
    with pytest.raises(ValueError, match="histogram_high_percent"):
        node.histogram_high_percent = 9.0
    with pytest.raises(ValueError, match="histogram_high_percent"):
        node.histogram_high_percent = 101.0
    with pytest.raises(ValueError, match="non-negative"):
        node.adaptation_speed_up = -1.0
    with pytest.raises(ValueError, match="min_exposure_ev"):
        node.min_exposure_ev = 17.0
    with pytest.raises(ValueError, match="max_exposure_ev"):
        node.max_exposure_ev = -17.0
    with pytest.raises(ValueError):
        node.auto_exposure_mode = 99  # type: ignore[assignment]
    with pytest.raises(ValueError):
        node.operator = 99  # type: ignore[assignment]

    node.auto_exposure = True
    with pytest.raises(ValueError, match="delta_time"):
        node(_constant_tensor(device, 0.25), delta_time=-1.0)

    scalar_input = spy.Tensor.empty(device, shape=(2, 2), dtype=float)
    with pytest.raises(ValueError, match="RGB or RGBA"):
        node(scalar_input)


def test_tonemapper_rejects_input_from_another_device() -> None:
    """Auto-exposure rejects containers whose storage belongs to another device."""
    if not helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("No GPU backend available")

    from falcor2.rendernodes import TonemapperNode

    device_type = helpers.DEFAULT_DEVICE_TYPES[0]
    node_device = helpers.get_device(device_type)
    input_device = helpers.get_device(device_type, use_cache=False)
    node = TonemapperNode.create(node_device)
    node.auto_exposure = True
    input_tensor = _constant_tensor(input_device, 0.25)

    with pytest.raises(ValueError, match="TonemapperNode device"):
        node(input_tensor)
