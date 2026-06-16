# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import falcor2 as f2
import numpy as np
import pytest
from falcor2.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32])
@pytest.mark.parametrize("element_count", [1, 123, 1234, 12345, 2000001])
@pytest.mark.parametrize("with_offset", [False, True])
@pytest.mark.parametrize("with_total_sum", [False, True])
def test_prefix_sum(
    device_type: spy.DeviceType,
    dtype: np.dtype,
    element_count: int,
    with_offset: bool,
    with_total_sum: bool,
):
    device = helpers.get_device(device_type)

    is_float = np.dtype(dtype).kind == "f"
    is_signed_integer = np.dtype(dtype).kind == "i"

    spy_type = {
        np.int32: spy.DataType.int32,
        np.int64: spy.DataType.int64,
        np.uint32: spy.DataType.uint32,
        np.uint64: spy.DataType.uint64,
        np.float32: spy.DataType.float32,
        np.float64: spy.DataType.float64,
    }[dtype]

    # Prepare random data.
    if is_float:
        data = np.random.random(element_count).astype(dtype)
    else:
        data = np.random.randint(
            -100 if is_signed_integer else 0, 100, size=element_count, dtype=dtype
        )
    # To test buffer offset, we pad the input on the left.
    if with_offset:
        data = np.hstack([np.array([0, 0, 0, 0], dtype=dtype), data])

    # Compute expected results.
    cumsum = np.cumsum(data, dtype=dtype)
    expected = np.hstack([np.array([0], dtype=dtype), cumsum[0:-1]])
    expected_sum = cumsum[-1]

    data_buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data,
    )

    total_sum_buffer = device.create_buffer(
        size=32,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    command_encoder = device.create_command_encoder()

    data_arg = data_buffer
    if with_offset:
        data_arg = {"buffer": data_buffer, "offset": 16}
    else:
        data_arg = data_buffer

    total_sum_arg = None
    if with_total_sum:
        if with_offset:
            total_sum_arg = {"buffer": total_sum_buffer, "offset": np.dtype(dtype).itemsize}
        else:
            total_sum_arg = total_sum_buffer

    f2.prefix_sum(
        command_encoder=command_encoder,
        type=spy_type,
        data=data_arg,
        element_count=element_count,
        total_sum=total_sum_arg,
    )

    device.submit_command_buffer(command_encoder.finish())

    result = data_buffer.to_numpy().view(dtype=dtype)
    if is_float:
        assert np.allclose(result, expected, rtol=1e-3)
    else:
        assert np.all(result == expected)

    if with_total_sum:
        result_sum = total_sum_buffer.to_numpy().view(dtype=dtype)[1 if with_offset else 0]
        if is_float:
            assert np.allclose(result_sum, expected_sum, rtol=1e-3)
        else:
            assert result_sum == expected_sum


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
