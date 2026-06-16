# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import falcor2 as f2
import numpy as np
import pytest
from falcor2.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32])
@pytest.mark.parametrize("element_count", [1, 100, 1234, 2000001])
@pytest.mark.parametrize("with_offset", [False, True])
def test_parallel_reduce_buffer(
    device: spy.Device,
    dtype: np.dtype,
    element_count: int,
    with_offset: bool,
):
    is_float = np.dtype(dtype).kind == "f"
    is_signed_integer = np.dtype(dtype).kind == "i"

    spy_type = {
        np.int32: spy.DataType.int32,
        np.uint32: spy.DataType.uint32,
        np.float32: spy.DataType.float32,
    }[dtype]

    # Prepare random data with a fixed seed for determinism.
    rng = np.random.default_rng(42)
    if is_float:
        data = rng.random(element_count).astype(dtype) * 2 - 1
    else:
        data = rng.integers(-100 if is_signed_integer else 0, 100, size=element_count, dtype=dtype)

    # To test buffer offset, we pad the input on the left.
    if with_offset:
        data = np.hstack([np.array([0, 0, 0, 0], dtype=dtype), data])

    # Compute expected results.
    actual_data = data[4:] if with_offset else data
    expected_sum = np.sum(actual_data, dtype=dtype)
    expected_min = np.min(actual_data)
    expected_max = np.max(actual_data)

    data_buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data,
    )

    element_size = np.dtype(dtype).itemsize
    result_buffer = device.create_buffer(
        size=element_size * 3 + 32,  # Extra space for offset testing
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    command_encoder = device.create_command_encoder()

    data_arg = data_buffer
    if with_offset:
        data_arg = {"buffer": data_buffer, "offset": 4 * element_size}

    sum_arg = {"buffer": result_buffer, "offset": 0}
    min_arg = {"buffer": result_buffer, "offset": element_size}
    max_arg = {"buffer": result_buffer, "offset": element_size * 2}

    f2.parallel_reduce(
        command_encoder=command_encoder,
        type=spy_type,
        data=data_arg,
        element_count=element_count,
        sum=sum_arg,
        min=min_arg,
        max=max_arg,
    )

    device.submit_command_buffer(command_encoder.finish())

    result = result_buffer.to_numpy().view(dtype=dtype)
    result_sum = result[0]
    result_min = result[1]
    result_max = result[2]

    if is_float:
        assert np.allclose(
            result_sum, expected_sum, rtol=1e-2, atol=1e-5
        ), f"Sum mismatch: got {result_sum}, expected {expected_sum}"
        assert np.allclose(
            result_min, expected_min, rtol=1e-5
        ), f"Min mismatch: got {result_min}, expected {expected_min}"
        assert np.allclose(
            result_max, expected_max, rtol=1e-5
        ), f"Max mismatch: got {result_max}, expected {expected_max}"
    else:
        assert (
            result_sum == expected_sum
        ), f"Sum mismatch: got {result_sum}, expected {expected_sum}"
        assert (
            result_min == expected_min
        ), f"Min mismatch: got {result_min}, expected {expected_min}"
        assert (
            result_max == expected_max
        ), f"Max mismatch: got {result_max}, expected {expected_max}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("element_count", [1, 1234])
def test_parallel_reduce_buffer_64bit(
    device: spy.Device,
    dtype: np.dtype,
    element_count: int,
):
    is_signed_integer = np.dtype(dtype).kind == "i"

    spy_type = {
        np.int64: spy.DataType.int64,
        np.uint64: spy.DataType.uint64,
    }[dtype]

    rng = np.random.default_rng(42)
    data = rng.integers(-100 if is_signed_integer else 0, 100, size=element_count, dtype=dtype)

    expected_sum = np.sum(data, dtype=dtype)
    expected_min = np.min(data)
    expected_max = np.max(data)

    data_buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data,
    )

    element_size = np.dtype(dtype).itemsize
    result_buffer = device.create_buffer(
        size=element_size * 3,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    command_encoder = device.create_command_encoder()

    sum_arg = {"buffer": result_buffer, "offset": 0}
    min_arg = {"buffer": result_buffer, "offset": element_size}
    max_arg = {"buffer": result_buffer, "offset": element_size * 2}

    f2.parallel_reduce(
        command_encoder=command_encoder,
        type=spy_type,
        data=data_buffer,
        element_count=element_count,
        sum=sum_arg,
        min=min_arg,
        max=max_arg,
    )

    device.submit_command_buffer(command_encoder.finish())

    result = result_buffer.to_numpy().view(dtype=dtype)
    assert result[0] == expected_sum, f"Sum mismatch: got {result[0]}, expected {expected_sum}"
    assert result[1] == expected_min, f"Min mismatch: got {result[1]}, expected {expected_min}"
    assert result[2] == expected_max, f"Max mismatch: got {result[2]}, expected {expected_max}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_parallel_reduce_selective_ops(device: spy.Device):
    """Test that providing only a subset of outputs (e.g., just min) works."""
    data = np.array([5, 3, 8, 1, 7, 2, 9, 4], dtype=np.int32)
    expected_min = np.min(data)

    data_buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data,
    )

    result_buffer = device.create_buffer(
        size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    command_encoder = device.create_command_encoder()

    f2.parallel_reduce(
        command_encoder=command_encoder,
        type=spy.DataType.int32,
        data=data_buffer,
        element_count=len(data),
        min=result_buffer,
    )

    device.submit_command_buffer(command_encoder.finish())

    result = result_buffer.to_numpy().view(dtype=np.int32)
    assert result[0] == expected_min


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_parallel_reduce_rejects_misaligned_data_offset(device: spy.Device):
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    data_buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data,
    )
    result_buffer = device.create_buffer(
        size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    command_encoder = device.create_command_encoder()

    with pytest.raises(Exception, match="data offset must be aligned"):
        f2.parallel_reduce(
            command_encoder=command_encoder,
            type=spy.DataType.int32,
            data={"buffer": data_buffer, "offset": 1},
            element_count=1,
            sum=result_buffer,
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_parallel_reduce_rejects_null_output_buffer(device: spy.Device):
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    data_buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data,
    )
    command_encoder = device.create_command_encoder()

    with pytest.raises(Exception, match="sum buffer is too small"):
        f2.parallel_reduce(
            command_encoder=command_encoder,
            type=spy.DataType.int32,
            data=data_buffer,
            element_count=len(data),
            sum=spy.BufferOffsetPair(),
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_parallel_reduction_texture_execute_overload(device: spy.Device):
    data = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    texture = device.create_texture(
        format=spy.Format.rgba32_float,
        width=1,
        height=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=data,
    )
    sum_buffer = device.create_buffer(
        size=16,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    command_encoder = device.create_command_encoder()

    reduction = f2.ParallelReduction(device)
    reduction.execute(command_encoder=command_encoder, input=texture, sum=sum_buffer)

    device.submit_command_buffer(command_encoder.finish())
    result_sum = sum_buffer.to_numpy().view(dtype=np.float32)
    assert np.allclose(result_sum, data.reshape(4), rtol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "texture_format,np_dtype",
    [
        (spy.Format.rgba32_float, np.float32),
        (spy.Format.r32_float, np.float32),
        (spy.Format.rgba32_uint, np.uint32),
        (spy.Format.rgba32_sint, np.int32),
    ],
)
@pytest.mark.parametrize(
    "width,height",
    [(1, 1), (64, 64), (100, 75)],
)
def test_parallel_reduce_texture(
    device: spy.Device,
    texture_format: spy.Format,
    np_dtype: np.dtype,
    width: int,
    height: int,
):
    format_info = spy.get_format_info(texture_format)
    channel_count = format_info.channel_count

    # Generate random data.
    if np.dtype(np_dtype).kind == "f":
        data = np.random.random((height, width, channel_count)).astype(np_dtype)
    elif np.dtype(np_dtype).kind == "i":
        data = np.random.randint(-50, 50, size=(height, width, channel_count), dtype=np_dtype)
    else:
        data = np.random.randint(0, 100, size=(height, width, channel_count), dtype=np_dtype)

    # Create texture.
    texture = device.create_texture(
        format=texture_format,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=data,
    )

    # Compute expected vec4 results.
    # Pad to 4 channels for comparison.
    padded = np.zeros((height, width, 4), dtype=np_dtype)
    padded[:, :, :channel_count] = data

    # Expected results (per channel).
    expected_sum = padded.reshape(-1, 4).sum(axis=0).astype(np_dtype)
    expected_min = padded.reshape(-1, 4).min(axis=0).astype(np_dtype)
    expected_max = padded.reshape(-1, 4).max(axis=0).astype(np_dtype)

    # For min/max, unused channels should have identity values from the shader.
    # We only check the valid channels.

    result_size = 16  # vec4 = 16 bytes
    sum_buffer = device.create_buffer(
        size=result_size,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    min_buffer = device.create_buffer(
        size=result_size,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    max_buffer = device.create_buffer(
        size=result_size,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    command_encoder = device.create_command_encoder()

    f2.parallel_reduce(
        command_encoder=command_encoder,
        input=texture,
        sum=sum_buffer,
        min=min_buffer,
        max=max_buffer,
    )

    device.submit_command_buffer(command_encoder.finish())

    result_sum = sum_buffer.to_numpy().view(dtype=np_dtype)
    result_min = min_buffer.to_numpy().view(dtype=np_dtype)
    result_max = max_buffer.to_numpy().view(dtype=np_dtype)

    if np.dtype(np_dtype).kind == "f":
        # Float sum can have significant accumulation error for large textures.
        rtol = 1e-2 if width * height > 100 else 1e-4
        assert np.allclose(
            result_sum[:channel_count], expected_sum[:channel_count], rtol=rtol, atol=1e-3
        ), f"Sum mismatch: got {result_sum[:channel_count]}, expected {expected_sum[:channel_count]}"
        assert np.allclose(
            result_min[:channel_count], expected_min[:channel_count], rtol=1e-5
        ), f"Min mismatch: got {result_min[:channel_count]}, expected {expected_min[:channel_count]}"
        assert np.allclose(
            result_max[:channel_count], expected_max[:channel_count], rtol=1e-5
        ), f"Max mismatch: got {result_max[:channel_count]}, expected {expected_max[:channel_count]}"
    else:
        assert np.all(
            result_sum[:channel_count] == expected_sum[:channel_count]
        ), f"Sum mismatch: got {result_sum[:channel_count]}, expected {expected_sum[:channel_count]}"
        assert np.all(
            result_min[:channel_count] == expected_min[:channel_count]
        ), f"Min mismatch: got {result_min[:channel_count]}, expected {expected_min[:channel_count]}"
        assert np.all(
            result_max[:channel_count] == expected_max[:channel_count]
        ), f"Max mismatch: got {result_max[:channel_count]}, expected {expected_max[:channel_count]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
